#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SOURCE_DIR/build"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread \
  "$SOURCE_DIR/o6_fc04_source.cpp" \
  -o "$SOURCE_DIR/build/o6_fc04_source"
