#!/bin/bash
# Returns the head camera to r1-teleimager on the rectified stream, and keeps the headset view on 60004.
set -e
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo systemctl stop r1-head-hd.service
~/miniconda3/envs/teleimager/bin/python3.10 "$dir/switch_head_source.py" rectified
sudo mkdir -p /etc/systemd/system/r1-head-hd.service.d
printf '[Service]\nExecStart=\nExecStart=/home/unitree/miniconda3/envs/teleimager/bin/python3.10 -u head_hd_server.py --zmq-port 55558 --webrtc-ports 60004\n' | sudo tee /etc/systemd/system/r1-head-hd.service.d/rollback.conf > /dev/null
sudo systemctl daemon-reload
sudo systemctl restart r1-teleimager.service
sudo systemctl start r1-head-hd.service
sleep 12
echo "r1-teleimager: $(systemctl is-active r1-teleimager.service)"
echo "r1-head-hd: $(systemctl is-active r1-head-hd.service)"
