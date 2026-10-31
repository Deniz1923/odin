"""Enforce upload bounds while receiving, including requests without Content-Length."""

from fastapi import HTTPException


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes=130_000_000):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        total = 0

        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                raise HTTPException(413, "Request exceeds upload limit")
            return message

        return await self.app(scope, bounded_receive, send)
