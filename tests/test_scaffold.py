import asyncio

from httpx2 import ASGITransport, AsyncClient

from face_surveillance.api.main import app


def test_health_and_openapi():
    # Same public contract without TestClient's cross-thread blocking portal.
    async def exercise():
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
            assert (await client.get("/openapi.json")).status_code == 200

    asyncio.run(exercise())
