#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
sdk="$HOME/unitree_r1_dev/unitree_sdk2"
g++ -std=c++17 -O2 -Wall -Wextra main.cpp -o r1_camera_services \
  -I"$sdk/include" -I"$sdk/thirdparty/include" -I"$sdk/thirdparty/include/ddscxx" \
  -L"$sdk/lib/aarch64" -L"$sdk/thirdparty/lib/aarch64" \
  -lunitree_sdk2 -lddscxx -lddsc -lpthread -lrt -Wl,-rpath,"$sdk/thirdparty/lib/aarch64"
