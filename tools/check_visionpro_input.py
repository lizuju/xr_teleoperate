import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop.utils.visionpro_source import VisionProMotionSource


def summarize(rows):
    metrics = {}
    paths = [(key, (key,)) for key in ("receive_age_ms", "source_age_ms", "left_age_ms", "right_age_ms")]
    paths += [(key, ("transport", key)) for key in (
        "upstream_age_ms", "receive_gap_ms", "source_interval_ms", "bridge_pending_ms",
        "bridge_prepare_ms", "bridge_to_receiver_ms", "receiver_json_ms", "receiver_processing_ms",
        "previous_snapshot_write_ms")]
    for name, path in paths:
        values = []
        seen = set()
        for row in rows:
            if len(path) > 1:
                identity = (row.get("stream_id"), row.get("transport", {}).get("packets_received"))
                if identity[1] is None or identity in seen:
                    continue
                seen.add(identity)
            value = row
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.append(value)
        values.sort()
        metrics[name] = {"samples": len(values), **({
            "p50_ms": values[math.ceil(.5*len(values))-1],
            "p95_ms": values[math.ceil(.95*len(values))-1], "max_ms": values[-1],
            "over_100ms": sum(value > 100. for value in values),
            "over_500ms": sum(value > 500. for value in values)} if values else {})}
    streams = {}
    first = {}
    for row in rows:
        if row.get("transport"):
            stream_id = str(row["stream_id"])
            first.setdefault(stream_id, row)
            streams[stream_id] = row
    return {"schema": "r1_tracking_diagnostic_v1", "samples": len(rows), "metrics": metrics,
            "hold_reason_samples": dict(Counter(row.get("hold_reason") or "none" for row in rows)),
            "last_transport_by_stream": {key: row["transport"] for key, row in streams.items()},
            "source_loss_delta_by_stream": {
                key: {side: row[side+"_loss_seq"]-first[key][side+"_loss_seq"]
                      for side in ("head", "left", "right") if side+"_loss_seq" in row and side+"_loss_seq" in first[key]}
                for key, row in streams.items()},
            "hand_missing_fraction": {side: sum(row.get(side+"_tracking") is False for row in rows)/len(rows)
                                      if rows else None for side in ("left", "right")},
            "notes": ["Pose ages use a minimum receive-minus-source clock estimate; baseline transport delay and clock drift are not independently measured.",
                      "upstream_age_ms includes sender/TCP/gRPC receive scheduling; it alone cannot prove Wi-Fi is the cause.",
                      "Receive gap and upstream maxima/counters include every bridge packet, including coalesced packets. Percentiles cover sampled packets only.",
                      "ss is captured on Ubuntu: its retrans fields describe Ubuntu outbound TCP, not all Vision Pro-to-Ubuntu retransmissions.",
                      "hold_reason is latched until realignment, so its sample count is not an interruption count."]}


def monitor_network(host, port, stop, output):
    with (output / "tcp.jsonl").open("w") as stream:
        while not stop.is_set():
            started = time.monotonic()
            result = subprocess.run(["ss", "-tin", "dst", host, "dport", "=", str(port)],
                                    capture_output=True, text=True, timeout=2.)
            stream.write(json.dumps({"monotonic": started, "wall_time_ns": time.time_ns(),
                                     "ss": result.stdout, "error": result.stderr,
                                     "returncode": result.returncode})+"\n")
            stream.flush()
            stop.wait(max(0., .2-(time.monotonic()-started)))


def main():
    parser = argparse.ArgumentParser(description="Read Vision Pro tracking only; does not import robot controllers or publish robot commands")
    parser.add_argument("host", help="Vision Pro Wi-Fi IP")
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--timeout", type=float, default=.5, help="Tracking freshness limit; match teleop (default 0.5 seconds)")
    parser.add_argument("--output", type=Path, help="New diagnostic directory for 50 Hz samples, TCP, ping and summary; never overwrites")
    parser.add_argument("--python", default=str(Path(__file__).resolve().parents[2] / ".venv-visionpro/bin/python"))
    args = parser.parse_args()
    if not 0. < args.seconds <= 60.:
        parser.error("--seconds must be between 0 and 60")
    if not 0. < args.timeout <= .5:
        parser.error("--timeout must be between 0 and 0.5 seconds")
    stop = threading.Event()
    workers, pings, handles, rows = [], [], [], []
    source = None
    samples = None
    seen_both = False
    failure = None
    try:
        if args.output:
            args.output.mkdir(parents=True, exist_ok=False)
            environment = {}
            for label, command in (
                ("route", ["ip", "-j", "route", "get", args.host]),
                ("default_route", ["ip", "-j", "route", "show", "default"]),
                ("wifi", ["nmcli", "-f", "IN-USE,SSID,CHAN,RATE,SIGNAL", "device", "wifi", "list", "--rescan", "no"]),
            ):
                result = subprocess.run(command, capture_output=True, text=True, timeout=3.)
                environment[label] = {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
            environment.update(host=args.host, port=args.port, tracking_timeout=args.timeout,
                               started_wall_ns=time.time_ns(), started_monotonic=time.monotonic())
            (args.output/"environment.json").write_text(json.dumps(environment, indent=2))
            targets = {"headset": args.host}
            routes = json.loads(environment["default_route"]["stdout"] or "[]")
            route = json.loads(environment["route"]["stdout"] or "[]")
            for entry in routes:
                if route and entry.get("dev") == route[0].get("dev") and entry.get("gateway"):
                    targets["gateway"] = entry["gateway"]
                    break
            for label, host in targets.items():
                handle = (args.output/(label+"-ping.log")).open("w")
                handles.append(handle)
                pings.append(subprocess.Popen(["ping", "-D", "-n", "-i", "0.2", "-W", "1", "-c",
                                               str(math.ceil((args.seconds+10)*5)), host], stdout=handle, stderr=subprocess.STDOUT))
            worker = threading.Thread(target=monitor_network, args=(args.host, args.port, stop, args.output))
            worker.start()
            workers.append(worker)
            samples = (args.output/"tracking.jsonl").open("w")
        source = VisionProMotionSource(args.host, args.python, args.port, args.timeout)
        deadline = time.monotonic() + args.seconds
        next_print = 0.
        while time.monotonic() < deadline:
            diagnostics = source.get_tracking_diagnostics()
            diagnostics.update(observed_monotonic=time.monotonic(), wall_time_ns=time.time_ns())
            seen_both |= diagnostics["left_tracking"] and diagnostics["right_tracking"]
            if samples:
                rows.append(diagnostics)
                samples.write(json.dumps(diagnostics, ensure_ascii=False)+"\n")
            if time.monotonic() >= next_print:
                print(json.dumps(diagnostics, ensure_ascii=False), flush=True)
                next_print = time.monotonic()+.5
            stop.wait(.02 if samples else .5)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        failure = str(exc)
        print(f"Read-only diagnostic failed: {failure}", file=sys.stderr)
    finally:
        if source is not None:
            source.close()
        stop.set()
        for worker in workers:
            worker.join(timeout=3.)
        for process in pings:
            if process.poll() is None:
                process.send_signal(2)
            process.wait(timeout=3.)
        for handle in handles:
            handle.close()
        if samples:
            samples.close()
            report = summarize(rows)
            report.update(seen_both_hands=seen_both, error=failure)
            (args.output/"summary.json").write_text(json.dumps(report, indent=2))
            print(f"Diagnostic saved to {args.output}")
    if failure or not seen_both:
        print("No successful valid-hand diagnostic completed; robot following has not been enabled.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
