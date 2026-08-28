#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root:"
  echo "  sudo bash scripts/setup_nvidia_container_toolkit_wsl.sh"
  exit 1
fi

if ! grep -qi microsoft /proc/version; then
  echo "Warning: this does not look like WSL2. Continuing with generic Ubuntu/Debian setup." >&2
else
  echo "Detected WSL2. Do not install a Linux NVIDIA display driver inside WSL."
fi

apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg2

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o "${tmpdir}/gpgkey"
gpg --dearmor -o "${tmpdir}/nvidia-container-toolkit-keyring.gpg" "${tmpdir}/gpgkey"
install -m 0755 -d /usr/share/keyrings
install -m 0644 "${tmpdir}/nvidia-container-toolkit-keyring.gpg" \
  /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  -o "${tmpdir}/nvidia-container-toolkit.list"
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  "${tmpdir}/nvidia-container-toolkit.list" \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit

nvidia-ctk runtime configure --runtime=docker

if command -v systemctl >/dev/null 2>&1 && systemctl is-system-running >/dev/null 2>&1; then
  systemctl restart docker
elif command -v service >/dev/null 2>&1; then
  service docker restart
else
  echo "Could not restart Docker automatically."
  echo "Restart the WSL Docker daemon, or run this from Windows:"
  echo "  wsl --shutdown"
fi

echo "Done. Verify with:"
echo "  docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi"
