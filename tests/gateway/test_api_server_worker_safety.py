"""Worker safety claims on the actual API health and capabilities routes."""

import types

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _app(adapter):
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_worker_safety_on_serving_health_and_capabilities(enabled):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter.gateway_runner = types.SimpleNamespace(control_only=False, _no_respawn_enabled=enabled)
    expected = {"protocol_version": 1, "control_only_recovery": True,
                "no_respawn_serving": True, "no_respawn_enabled": enabled}
    async with TestClient(TestServer(_app(adapter))) as cli:
        for path in ("/health", "/v1/health", "/health/detailed", "/v1/capabilities"):
            response = await cli.get(path)
            assert response.status == 200
            assert (await response.json())["worker_safety"] == expected


@pytest.mark.asyncio
async def test_control_health_advertises_mode_without_opening_capabilities():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "secret"}))
    adapter.gateway_runner = types.SimpleNamespace(control_only=True, _no_respawn_enabled=True)
    expected = {"protocol_version": 1, "control_only_recovery": True,
                "no_respawn_serving": True, "no_respawn_enabled": True}
    async with TestClient(TestServer(_app(adapter))) as cli:
        assert (await cli.get("/health")).status == 401
        for path in ("/health", "/v1/health", "/health/detailed"):
            response = await cli.get(path, headers={"Authorization": "Bearer secret"})
            assert response.status == 200
            assert (await response.json())["worker_safety"] == expected
        response = await cli.get("/v1/capabilities", headers={"Authorization": "Bearer secret"})
        assert response.status == 503
