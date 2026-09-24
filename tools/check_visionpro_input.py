import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop.utils.visionpro_source import VisionProMotionSource


def main():
    parser = argparse.ArgumentParser(description="Read Vision Pro tracking only; does not import robot controllers or publish robot commands")
    parser.add_argument("host", help="Vision Pro Wi-Fi IP")
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--python", default=str(Path(__file__).resolve().parents[2] / ".venv-visionpro/bin/python"))
    args = parser.parse_args()
    if not 0. < args.seconds <= 60.:
        parser.error("--seconds must be between 0 and 60")
    source = VisionProMotionSource(args.host, args.python, args.port)
    seen_both = False
    try:
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            diagnostics = source.get_tracking_diagnostics()
            seen_both |= diagnostics["left_tracking"] and diagnostics["right_tracking"]
            print(json.dumps(diagnostics, ensure_ascii=False), flush=True)
            time.sleep(0.5)
    finally:
        source.close()
    if not seen_both:
        print("No valid pair of hands observed; robot following has not been enabled.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
