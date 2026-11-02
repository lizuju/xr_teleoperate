"""Read-only WebRTC viewer that measures what a headset would receive from head_hd_server."""
import asyncio, json, ssl, sys, time
import aiohttp, cv2
from aiortc import RTCPeerConnection, RTCSessionDescription
url, seconds, snapshot = sys.argv[1], float(sys.argv[2]), sys.argv[3]
async def main():
    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")
    channel = pc.createDataChannel("r1-video-meta-v1", ordered=False, maxRetransmits=0)
    meta, latency = {}, []
    def on_message(m):
        if '"frame"' in m:
            info = json.loads(m); meta[info["rtp_timestamp"]] = info
            latency.append((info["send_monotonic_ns"] - info["source_monotonic_ns"]) / 1e6)
    channel.on("message", on_message)
    frames, sizes, frame_bytes = [], set(), []
    done = asyncio.Event()
    @pc.on("track")
    def on_track(track):
        async def run():
            end = None
            saved = False
            while True:
                frame = await track.recv()
                now = time.monotonic_ns()
                end = end or time.monotonic() + seconds
                frames.append(now); sizes.add((frame.width, frame.height))
                if len(frames) % 30 == 7:
                    frame_bytes.append(cv2.Laplacian(frame.to_ndarray(format="gray"), cv2.CV_64F).var())
                if not saved and len(frames) > 200:
                    cv2.imwrite(snapshot, frame.to_ndarray(format="bgr24")); saved = True
                if time.monotonic() > end: break
            done.set()
        asyncio.ensure_future(run())
    await pc.setLocalDescription(await pc.createOffer())
    while pc.iceGatheringState != "complete": await asyncio.sleep(0.02)
    context = ssl.create_default_context(); context.check_hostname = False; context.verify_mode = ssl.CERT_NONE
    async with aiohttp.ClientSession() as session:
        async with session.post(url, ssl=context, json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type,
                                                        "codec": "h264", "video_metadata": "r1-video-meta-v1"}) as r:
            answer = await r.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    started = time.monotonic()
    while not done.is_set() and time.monotonic() - started < seconds + 25:
        await asyncio.sleep(5)
        print(f"t={time.monotonic() - started:.0f}s ice={pc.iceConnectionState} conn={pc.connectionState} frames={len(frames)} metadata={len(meta)} channel={channel.readyState}", flush=True)
    if not frames:
        print("no frames received"); await pc.close(); return
    gaps = sorted((b - a) / 1e6 for a, b in zip(frames, frames[1:]))
    big = [(round((b - frames[0]) / 1e9, 1), round((b - a) / 1e6)) for a, b in zip(frames, frames[1:]) if b - a > 150e6]
    print("viewer gaps over 150 ms (time s, gap ms):", big[:12])
    sent = [m["send_monotonic_ns"] for m in meta.values()]
    source = [m["source_monotonic_ns"] for m in meta.values()]
    for label, stamps in (("server rtp send", sent), ("camera source", source)):
        stamps.sort()
        d = sorted((b - a) / 1e6 for a, b in zip(stamps, stamps[1:]))
        print(label, "interval_ms p50", round(d[len(d)//2], 1), "p99", round(d[int(len(d)*.99)], 1), "max", round(d[-1], 1), "over 150 ms:", sum(1 for x in d if x > 150))
    stats = await pc.getStats()
    inbound = [s for s in stats.values() if s.type == "inbound-rtp"]
    latency.sort()
    print("detail (Laplacian var) p50", sorted(frame_bytes)[len(frame_bytes)//2] if frame_bytes else None)
    print("sizes", sizes, "frames", len(frames), "fps", round((len(frames) - 1) / ((frames[-1] - frames[0]) / 1e9), 2))
    print("interval_ms p50", round(gaps[len(gaps)//2], 1), "p99", round(gaps[int(len(gaps)*.99)], 1), "max", round(gaps[-1], 1))
    print("metadata frames", len(meta), "source_to_rtp_sent_ms p50", round(latency[len(latency)//2], 1) if latency else None, "p99", round(latency[int(len(latency)*.99)], 1) if latency else None)
    for s in inbound:
        print("rtp packets", s.packetsReceived, "lost", s.packetsLost, "jitter", s.jitter)
    await pc.close()
asyncio.run(main())
