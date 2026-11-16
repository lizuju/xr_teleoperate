import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


spec = importlib.util.spec_from_file_location(
    "video_gateway_identity",
    Path(__file__).resolve().parents[1] / "deploy/visionpro-video/r1_webrtc_gateway.py",
)
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)


class UpstreamResponse:
    status = 200
    headers = {"Content-Type": "application/json"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def read(self):
        return b'{"sdp":"answer","type":"answer"}'


class GatewayIdentityTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.forwarded = []
        self.remote = None

        class Session:
            def request(session, method, url, **kwargs):
                self.forwarded.append({"method": method, "url": url, **kwargs})
                return UpstreamResponse()

        async def proxy(request):
            if self.remote is not None:
                request = request.clone(remote=self.remote)
            return await gateway.proxy(request)

        app = web.Application()
        app[gateway.SESSION] = Session()
        app[gateway.SETTINGS] = {
            "bind_port": 60004,
            "bind_host": "192.168.50.165",
            "upstream": "https://192.168.123.164:60004",
        }
        app.router.add_post("/offer", proxy)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def offer(self, payload, headers=None):
        async with self.client.post("/offer", json=payload, headers=headers) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(await response.json(), {"sdp": "answer", "type": "answer"})
        return json.loads(self.forwarded[-1]["data"])

    async def test_native_identity_uses_connection_address_over_client_fields(self):
        payload = {
            "sdp": "v=0\r\n", "type": "offer", "codec": "h264",
            "video_metadata": "r1-video-meta-v1", "native_client_id": "another-headset",
        }
        forwarded = await self.offer(payload, headers={"X-Forwarded-For": "192.168.50.99"})
        self.assertEqual(forwarded, {**payload, "native_client_id": "127.0.0.1"})

    async def test_headsets_are_distinct_and_reconnect_keeps_identity(self):
        payload = {"sdp": "v=0\r\n", "type": "offer", "video_metadata": "r1-video-meta-v1"}
        identities = []
        for remote in ("192.168.50.83", "192.168.50.84", "192.168.50.83"):
            self.remote = remote
            identities.append((await self.offer(payload))["native_client_id"])
        self.assertEqual(identities, ["192.168.50.83", "192.168.50.84", "192.168.50.83"])

    async def test_safari_offer_is_unchanged(self):
        payload = {"sdp": "v=0\r\na=candidate:1 1 udp 123 192.168.50.83 5000 typ host\r\n",
                   "type": "offer", "codec": "h264", "extra": {"value": 1}}
        forwarded = await self.offer(payload, headers={"User-Agent": "Version/26.0 Safari/605.1.15"})
        self.assertEqual(forwarded, payload)
        self.assertNotIn("native_client_id", forwarded)
        self.assertEqual(self.forwarded[-1]["headers"]["User-Agent"], "Version/26.0 Safari/605.1.15")

    async def test_other_metadata_protocol_does_not_enable_replacement(self):
        for marker in (None, "r1-video-meta-v2", "other-client"):
            with self.subTest(marker=marker):
                payload = {"sdp": "v=0\r\n", "type": "offer", "video_metadata": marker,
                           "native_client_id": "client-owned-field"}
                self.assertEqual(await self.offer(payload), payload)

    async def test_native_identity_preserves_mdns_resolution(self):
        self.remote = "192.168.50.83"
        sdp = ("v=0\r\na=candidate:1 1 udp 123 headset.local 5000 typ host\r\n"
               "a=candidate:2 1 udp 122 headset.local 5001 typ host\r\n"
               "a=end-of-candidates\r\n")
        payload = {"sdp": sdp, "type": "offer", "video_metadata": "r1-video-meta-v1"}
        with patch.object(gateway, "resolve_mdns", new=AsyncMock(return_value=self.remote)) as resolve:
            forwarded = await self.offer(payload)
        resolve.assert_awaited_once_with("headset.local")
        self.assertEqual(forwarded, {**payload, "sdp": sdp.replace("headset.local", self.remote),
                                     "native_client_id": self.remote})


if __name__ == "__main__":
    unittest.main()
