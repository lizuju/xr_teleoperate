#!/bin/bash
# Restarts the camera services so r1-head-hd publishes the head camera. Run once; asks for the sudo password.
set -e
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo install -m 644 "$dir/r1-head-hd.service" /etc/systemd/system/r1-head-hd.service
# A rollback leaves this override, which would publish on the wrong ZMQ port.
sudo rm -f /etc/systemd/system/r1-head-hd.service.d/rollback.conf
sudo systemctl daemon-reload
sudo systemctl stop r1-head-hd.service
sudo systemctl restart r1-teleimager.service
sudo systemctl start r1-head-hd.service
sleep 12
echo "r1-teleimager: $(systemctl is-active r1-teleimager.service)"
echo "r1-head-hd: $(systemctl is-active r1-head-hd.service)"
journalctl -u r1-head-hd.service --no-pager -n 12 | grep -E '\[HeadHD\]' | tail -2 | sed -E 's/.*\]: //'
