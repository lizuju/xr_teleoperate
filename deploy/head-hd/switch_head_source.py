"""Move the head camera from r1-teleimager to r1-head-hd, or back.

Edits the r1-teleimager config in place and keeps a backup beside it. The services
read it on restart, so run this before restarting them. Takes no action on the
running services. Needs teleimager with external_publisher support.

Usage: switch_head_source.py raw | rectified
"""
import shutil
import sys
from pathlib import Path

import yaml

CONFIG = Path.home() / ".config/teleimager/teleimager_server.yaml"
BACKUP = CONFIG.with_name(CONFIG.name + ".pre-raw-head")
RAW_HEAD = {"type": "rtp_jpeg_stereo", "left_rtp_port": 5004, "right_rtp_port": 5005,
            "image_shape": [544, 1280], "fps": 30, "external_publisher": True}


def to_raw():
    config = yaml.safe_load(CONFIG.read_text())
    head = config["camera"]["head_camera"]
    if head.get("type") != "rtp_h264_stereo" or head.get("image_shape") != [448, 1088]:
        sys.exit(f"ABORT: unexpected head camera config: {head}")
    if BACKUP.exists():
        sys.exit(f"ABORT: backup already exists: {BACKUP}")
    shutil.copy2(CONFIG, BACKUP)
    head.update(RAW_HEAD)
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
    print("r1-teleimager now leaves the head camera to r1-head-hd")


def to_rectified():
    if not BACKUP.exists():
        sys.exit(f"ABORT: no backup: {BACKUP}")
    shutil.move(str(BACKUP), str(CONFIG))
    print("r1-teleimager config restored")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("raw", "rectified"):
        sys.exit(__doc__)
    (to_raw if sys.argv[1] == "raw" else to_rectified)()
