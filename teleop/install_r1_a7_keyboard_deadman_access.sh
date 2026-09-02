#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run with sudo: sudo bash $0" >&2
  exit 1
fi

usermod -aG input hnh
echo "Added hnh to input. Log out and back in, or reboot Ubuntu, before starting the deadman."
