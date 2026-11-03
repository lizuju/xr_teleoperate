#!/usr/bin/env bash
# Runs r1_camera_services at every PC2 boot. Remove with:
#   sudo systemctl disable r1-camera-services.service
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
sudo install -m 644 r1-camera-services.service /etc/systemd/system/r1-camera-services.service
sudo systemctl daemon-reload
sudo systemctl enable r1-camera-services.service
systemctl is-enabled r1-camera-services.service
