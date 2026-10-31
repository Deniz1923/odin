import asyncio
import io
import time

import cv2
import numpy as np
import pytest
from PIL import Image
from skimage import data


def photo():
    stream = io.BytesIO()
    Image.fromarray(data.astronaut()).save(stream, format="PNG")
    return stream.getvalue()


def test_enrollment_is_atomic_and_delete_invalidates_worker(tmp_path):
    from face_surveillance.classical.service import Service

    service = Service(tmp_path / "private")
    with pytest.raises(ValueError):
        service.enroll("A", [photo(), b"not an image"], True, True)
    assert service.store.people() == []
    person = service.enroll("A", [photo()], True, True)
    assert service.store.people()[0]["samples"] == 1
    assert list((tmp_path / "private").glob("*.png")) == []
    service.delete(person["id"])
    assert service.store.gallery() == {}
    service.close()


def test_recorded_source_finishes_and_deletes_temporary_video(tmp_path):
    from face_surveillance.classical.service import Service

    service = Service(tmp_path / "private")
    video = service.store.directory / "upload.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()
    service.start(str(video), "replay:test", owned=True)
    deadline = time.monotonic() + 5
    while service.status()["state"] not in ("ended", "failed") and time.monotonic() < deadline:
        time.sleep(0.03)
    assert service.status()["state"] == "ended"
    assert service.status()["processed"] == 3
    assert service.status()["dropped"] == 0
    assert not video.exists()
    assert service.preview() is None
    service.close()


def test_api_requires_mutation_token_and_supports_gallery_workflow(tmp_path):
    from httpx2 import ASGITransport, AsyncClient

    from face_surveillance.api.main import create_app

    async def exercise():
        app = create_app(tmp_path / "private")
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                assert (await client.get("/health")).json() == {"status": "ok"}
                assert (await client.post("/api/stop")).status_code == 403
                token = (await client.get("/api/session")).json()["token"]
                headers = {"Authorization": f"Bearer {token}"}
                bad = await client.post(
                    "/api/people",
                    data={"name": "A", "authorized": "true"},
                    files=[("images", ("bad.png", b"bad", "image/png"))],
                    headers=headers,
                )
                assert bad.status_code == 422
                assert (await client.get("/api/people")).json() == []
                response = await client.post(
                    "/api/people",
                    data={"name": "A", "authorized": "true"},
                    files=[("images", ("a.png", photo(), "image/png"))],
                    headers=headers,
                )
                assert response.status_code == 201, response.text
                person = response.json()
                assert (
                    await client.delete("/api/people/" + person["id"], headers=headers)
                ).status_code == 204
                assert (await client.get("/api/people")).json() == []
                blocked = await client.post(
                    "/api/stop", headers={**headers, "Origin": "https://evil.example"}
                )
                assert blocked.status_code == 403

    asyncio.run(exercise())


def test_policy_survives_restart_and_gallery_reload_does_not_reuse_track_ids(tmp_path):
    from face_surveillance.classical.pipeline import Policy
    from face_surveillance.classical.service import Service

    path = tmp_path / "private"
    service = Service(path)
    service.configure(Policy(threshold=0.123))
    service.pipeline.next_id = 17
    service.enroll("A", [photo()], True)
    assert service.pipeline.next_id >= 17
    service.close()
    restarted = Service(path)
    assert restarted.policy.threshold == 0.123
    restarted.close()


def test_busy_vision_lock_does_not_block_health_requests(tmp_path):
    import threading

    from httpx2 import ASGITransport, AsyncClient

    from face_surveillance.api.main import create_app

    async def exercise():
        app = create_app(tmp_path / "private")
        async with app.router.lifespan_context(app):
            locked = threading.Event()

            def occupy():
                with app.state.service.lock:
                    locked.set()
                    time.sleep(0.4)

            worker = threading.Thread(target=occupy)
            worker.start()
            assert locked.wait(1)
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                started = time.monotonic()
                status_task = asyncio.create_task(client.get("/api/status"))
                await asyncio.sleep(0.01)
                assert (await client.get("/health")).status_code == 200
                assert time.monotonic() - started < 0.2
                await status_task
            worker.join()

    asyncio.run(exercise())


def test_empty_video_fails_instead_of_appearing_complete(tmp_path):
    from face_surveillance.classical.service import Service

    service = Service(tmp_path / "private")
    video = tmp_path / "empty.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    writer.release()
    service.start(str(video), "empty")
    deadline = time.monotonic() + 3
    while service.status()["state"] not in ("ended", "failed") and time.monotonic() < deadline:
        time.sleep(0.02)
    assert service.status()["state"] == "failed"
    assert service.status()["error"]
    service.close()


def test_chunked_request_cannot_bypass_body_limit(tmp_path):
    from httpx2 import ASGITransport, AsyncClient

    from face_surveillance.api.main import create_app

    async def exercise():
        app = create_app(tmp_path / "private", max_body_bytes=1024)
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                token = (await client.get("/api/session")).json()["token"]

                async def chunks():
                    yield b" " * 600
                    yield b" " * 600

                response = await client.put(
                    "/api/policy",
                    content=chunks(),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )
                assert response.status_code == 413

    asyncio.run(exercise())


def test_api_replay_persists_event_and_identity_deletion_removes_it(tmp_path):
    from httpx2 import ASGITransport, AsyncClient

    from face_surveillance.api.main import create_app

    video = tmp_path / "known.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (512, 512))
    for _ in range(10):
        writer.write(cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR))
    writer.release()

    async def exercise():
        app = create_app(tmp_path / "private")
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                token = (await client.get("/api/session")).json()["token"]
                headers = {"Authorization": f"Bearer {token}"}
                person = (
                    await client.post(
                        "/api/people",
                        headers=headers,
                        data={"name": "Fixture", "authorized": "true", "alert_enabled": "true"},
                        files=[("images", ("fixture.png", photo(), "image/png"))],
                    )
                ).json()
                response = await client.post(
                    "/api/video",
                    headers=headers,
                    files={"file": ("known.avi", video.read_bytes(), "video/x-msvideo")},
                )
                assert response.status_code == 200, response.text
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    status = (await client.get("/api/status")).json()
                    if status["state"] in ("ended", "failed"):
                        break
                    await asyncio.sleep(0.05)
                assert status["state"] == "ended", status
                events = (await client.get("/api/events")).json()
                assert len(events) == 1
                assert events[0]["person_id"] == person["id"]
                assert not list((tmp_path / "private").glob("upload-*.video"))
                assert (await client.get("/api/preview")).status_code == 204
                assert (
                    await client.delete("/api/people/" + person["id"], headers=headers)
                ).status_code == 204
                assert (await client.get("/api/events")).json() == []

    asyncio.run(exercise())
