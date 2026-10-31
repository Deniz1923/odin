"""Local operator API; run one uvicorn worker bound to loopback."""

import asyncio
import hashlib
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from face_surveillance.api.limits import BodyLimitMiddleware
from face_surveillance.classical.detection import ROOT
from face_surveillance.classical.pipeline import Policy
from face_surveillance.classical.service import Service


class CameraRequest(BaseModel):
    camera: int = Field(default=0, ge=0, le=7)


class AlertRequest(BaseModel):
    enabled: bool


def create_app(data_dir: Path | None = None, *, max_body_bytes: int = 130_000_000) -> FastAPI:
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = Service(
            data_dir or Path(os.environ.get("ODIN_DATA_DIR", ROOT / "data/classical"))
        )
        try:
            yield
        finally:
            app.state.service.close()

    app = FastAPI(title="odin — Classical Face Recognition", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"]
    )
    app.add_middleware(BodyLimitMiddleware, max_bytes=max_body_bytes)

    @app.middleware("http")
    async def protect_local_mutations(request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            allowed = {
                str(request.base_url).rstrip("/"),
                "http://127.0.0.1:5173",
                "http://localhost:5173",
            }
            if (origin and origin not in allowed) or request.headers.get(
                "authorization"
            ) != f"Bearer {token}":
                return JSONResponse(
                    {"detail": "Local mutation token/origin rejected"}, status_code=403
                )
            try:
                length = int(request.headers.get("content-length", "0"))
            except ValueError:
                return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
            if length > max_body_bytes:
                return JSONResponse({"detail": "Request exceeds 130 MB"}, status_code=413)
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    async def command(function, *args):
        service = app.state.service
        if not service.command_slots.acquire(blocking=False):
            raise HTTPException(429, "Command queue full")
        future = service.executor.submit(function, *args)
        future.add_done_callback(lambda _: service.command_slots.release())
        while not future.done():
            await asyncio.sleep(0.02)
        try:
            return future.result()
        except KeyError as exc:
            raise HTTPException(404, "Record not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/session")
    async def session():
        return {"token": token}

    @app.get("/api/status")
    async def status():
        return await command(app.state.service.status)

    @app.get("/api/preview")
    async def preview():
        image = await command(app.state.service.preview)
        if image is None:
            return Response(status_code=204, headers={"Cache-Control": "no-store"})
        return Response(image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/people")
    async def people():
        return await command(app.state.service.store.people)

    @app.post("/api/people", status_code=201)
    async def enroll(request: Request):
        async with request.form(max_files=32, max_fields=4, max_part_size=8_000_000) as form:
            images, total = [], 0
            for file in form.getlist("images"):
                if not hasattr(file, "read"):
                    raise HTTPException(422, "Expected image uploads")
                content = await file.read(8_000_001)
                total += len(content)
                if len(content) > 8_000_000 or total > 64_000_000:
                    raise HTTPException(413, "Enrollment upload too large")
                images.append(content)
            return await command(
                app.state.service.enroll,
                str(form.get("name", "")),
                images,
                form.get("authorized") == "true",
                form.get("alert_enabled") == "true",
            )

    @app.delete("/api/people/{person_id}", status_code=204)
    async def delete(person_id: str):
        await command(app.state.service.delete, person_id)
        return Response(status_code=204)

    @app.patch("/api/people/{person_id}/alerts", status_code=204)
    async def set_alert(person_id: str, body: AlertRequest):
        await command(app.state.service.set_alert, person_id, body.enabled)
        return Response(status_code=204)

    @app.post("/api/start")
    async def start(body: CameraRequest):
        await command(app.state.service.start, body.camera, f"webcam:{body.camera}")
        return await command(app.state.service.status)

    @app.post("/api/video")
    async def video(file: UploadFile):
        service = app.state.service
        path = service.store.directory / f"upload-{uuid.uuid4().hex}.video"
        size, digest, started = 0, hashlib.sha256(), False
        try:
            with path.open("xb") as output:
                path.chmod(0o600)
                while chunk := await file.read(1_000_000):
                    size += len(chunk)
                    if size > 128_000_000:
                        raise HTTPException(413, "Video exceeds 128 MB")
                    digest.update(chunk)
                    output.write(chunk)
            if size == 0:
                raise HTTPException(422, "Empty video")
            await command(
                lambda: service.start(str(path), "replay:" + digest.hexdigest()[:12], owned=True)
            )
            started = True
            return await command(service.status)
        finally:
            await file.close()
            if not started:
                path.unlink(missing_ok=True)

    @app.post("/api/stop")
    async def stop():
        await command(app.state.service.stop)
        return await command(app.state.service.status)

    @app.put("/api/policy")
    async def policy(body: Policy):
        await command(app.state.service.configure, body)
        return body.model_dump()

    @app.get("/api/events")
    async def events():
        return await command(app.state.service.store.events)

    @app.post("/api/events/{event_id}/ack", status_code=204)
    async def acknowledge(event_id: str):
        await command(app.state.service.store.acknowledge, event_id)
        return Response(status_code=204)

    dist = ROOT / "frontend/dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="operator")
    return app


app = create_app()
