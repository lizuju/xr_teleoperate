#!/bin/bash
# Installs the service on PC2. Run once; it asks for the sudo password.
set -e
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo install -m 644 "$dir/r1-head-hd.service" /etc/systemd/system/r1-head-hd.service
sudo systemctl daemon-reload
sudo systemctl enable --now r1-head-hd.service
sleep 8
systemctl is-active r1-head-hd.service
journalctl -u r1-head-hd.service --no-pager -n 3
