import argparse
import asyncio
import ipaddress
import json
import logging
import os
import time
import re
import ssl
from pathlib import Path

import aiohttp
from aiohttp import web


# One gateway process per camera WebRTC port. PC2 (192.168.123.164) is on the
# robot's wired network and cannot be reached by the headset, which is on the
# 192.168.124.x WiFi, so each of PC2's teleimager WebRTC signalling ports gets a
# TLS reverse proxy on the Ubuntu host:
#
#   60001 -> PC2 60001  head stereo  (main_image_binocular_webrtc background)
#   60002 -> PC2 60002  left wrist   (independent HUD panel)
#   60003 -> PC2 60003  right wrist  (independent HUD panel)
#
# The proxy also rewrites mDNS (.local) ICE candidates in the SDP offer into the
# IPv4 addresses the headset can reach; PC2 advertises its own hostname.
DEFAULT_BIND_HOST = "192.168.124.147"
DEFAULT_UPSTREAM_HOST = "192.168.123.164"
UPSTREAM_PATH = "/offer"
CERT_DIR = Path.home() / ".config/xr_teleoperate"
CANDIDATE_HOST = re.compile(
    r"^(a=candidate:\S+ \d+ \S+ \d+ )([A-Za-z0-9][A-Za-z0-9.-]*\.local\.?)(?= )",
    re.MULTILINE | re.IGNORECASE,
)
CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}
SESSION = web.AppKey("upstream_session", aiohttp.ClientSession)
SETTINGS = web.AppKey("settings", dict)
logger = logging.getLogger("r1-webrtc-gateway")


async def resolve_mdns(hostname):
    process = await asyncio.create_subprocess_exec(
        "/usr/bin/timeout", "3", "/usr/bin/avahi-resolve-host-name", "-4", hostname,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    fields = stdout.decode().split()
    if process.returncode != 0 or len(fields) != 2:
        raise web.HTTPBadGateway(
            text=json.dumps({"error": f"mDNS resolution failed: {hostname}"}),
            content_type="application/json", headers=CORS,
        )
    try:
        address = str(ipaddress.IPv4Address(fields[1]))
    except ValueError as error:
        raise web.HTTPBadGateway(text="Invalid mDNS address", headers=CORS) from error
    logger.info("mDNS %s -> %s", hostname, address)
    return address


async def proxy(request):
    settings = request.app[SETTINGS]
    # Log every request, not just accepted offers: an XR page that never reaches
    # a gateway (wrong URL, TLS/CORS failure, element never mounted) looks
    # identical to "no offer" otherwise.
    logger.info("port %s: %s %s from %s", settings["bind_port"], request.method,
                request.path, request.remote)

    if request.method == "OPTIONS":
        return web.Response(headers=CORS)

    body = await request.read()
    headers = {"User-Agent": request.headers.get("User-Agent", "")}
    if request.method == "POST":
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError) as error:
            raise web.HTTPBadRequest(text="Invalid JSON", headers=CORS) from error
        if not isinstance(payload, dict) or not isinstance(payload.get("sdp"), str):
            raise web.HTTPBadRequest(text="Missing SDP string", headers=CORS)
        names = list(dict.fromkeys(match[2] for match in CANDIDATE_HOST.finditer(payload["sdp"])))
        addresses = await asyncio.gather(*(resolve_mdns(name) for name in names))
        resolved = dict(zip(names, addresses))
        payload["sdp"] = CANDIDATE_HOST.sub(
            lambda match: match[1] + resolved[match[2]], payload["sdp"]
        )
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
        logger.info("port %s: offer client=%s resolved_candidates=%d",
                    settings["bind_port"], request.remote, len(resolved))

    try:
        async with request.app[SESSION].request(
            request.method, settings["upstream"] + request.path, data=body,
            headers=headers, server_hostname=settings["bind_host"], allow_redirects=False,
        ) as response:
            response_headers = dict(CORS)
            if "Content-Type" in response.headers:
                response_headers["Content-Type"] = response.headers["Content-Type"]
            return web.Response(
                status=response.status, body=await response.read(), headers=response_headers,
            )
    except (aiohttp.ClientError, asyncio.TimeoutError) as error:
        logger.error("Upstream video service failed: %s", error)
        raise web.HTTPBadGateway(text="Upstream video service unavailable", headers=CORS) from error


def read_operator_status(now_ns=None):
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    path = Path(f"/tmp/r1-teleop-status-{os.getuid()}.json")
    try:
        with path.open("rb") as stream:
            raw = stream.read(32769)
        if len(raw) > 32768:
            raise ValueError("Status exceeds size limit")
        status = json.loads(raw)
        if (not isinstance(status, dict) or status.get("schema") != "r1_teleop_status_v1"
                or not isinstance(status.get("run_id"), str)
                or type(status.get("sequence")) is not int
                or type(status.get("sample_monotonic_ns")) is not int
                or status.get("motion") not in {"waiting", "preparing", "following", "paused", "tracking_hold", "stopped", "failed"}
                or not isinstance(status.get("recording"), dict)):
            raise ValueError("Invalid status")
    except FileNotFoundError:
        return {"available": False, "reason": "not_running"}
    except (OSError, ValueError, UnicodeDecodeError):
        return {"available": False, "reason": "invalid"}
    age_ns = now_ns - status["sample_monotonic_ns"]
    if not 0 <= age_ns <= 1_000_000_000:
        return {"available": False, "reason": "stale"}
    return {"available": True, "age_ms": age_ns / 1e6, "status": status}


async def operator_status(request):
    return web.json_response(read_operator_status(), headers={"Cache-Control": "no-store"})


async def upstream_session(app):
    context = ssl.create_default_context(cafile=str(CERT_DIR / "rootCA.pem"))
    # The upstream port is part of the TLS SNI-independent path only; the
    # certificate is the same self-signed root on every teleimager port.
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=context),
        timeout=aiohttp.ClientTimeout(total=15),
    ) as session:
        app[SESSION] = session
        yield


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", default=DEFAULT_BIND_HOST)
    parser.add_argument("--bind-port", type=int, default=60001)
    parser.add_argument("--upstream-host", default=DEFAULT_UPSTREAM_HOST)
    parser.add_argument("--upstream-port", type=int, default=None,
                        help="PC2 teleimager WebRTC port; defaults to --bind-port")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    upstream_port = args.upstream_port or args.bind_port
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    app = web.Application()
    app[SETTINGS] = {
        "bind_host": args.bind_host,
        "bind_port": args.bind_port,
        "upstream": f"https://{args.upstream_host}:{upstream_port}",
    }
    app.cleanup_ctx.append(upstream_session)
    app.router.add_get("/", proxy)
    app.router.add_get("/client.js", proxy)
    app.router.add_post(UPSTREAM_PATH, proxy)
    if args.bind_port == 60001:
        app.router.add_get("/r1/status", operator_status)
    for path in ("/", "/client.js", UPSTREAM_PATH):
        app.router.add_options(path, proxy)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_DIR / "cert.pem", CERT_DIR / "key.pem")
    logger.info("gateway %s:%s -> %s", args.bind_host, args.bind_port, app[SETTINGS]["upstream"])
    web.run_app(app, host=args.bind_host, port=args.bind_port,
                ssl_context=context, access_log=None)
