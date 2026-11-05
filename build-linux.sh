#!/usr/bin/env bash
# Build Speed Dreams on Debian/Ubuntu (also inside WSL2), then the rlpilot native layer and Python env.
# In WSL2, keep the repo on the Linux filesystem (~/...), not under /mnt/c.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SD="$ROOT/speed-dreams-code"; PREFIX="$ROOT/sd-install"

sudo apt-get update
sudo apt-get install -y git cmake build-essential libopenscenegraph-dev libcurl4-gnutls-dev libsdl2-dev \
  libsdl2-mixer-dev libsdl2-ttf-dev librhash-dev libenet-dev libpng-dev libjpeg-dev zlib1g-dev libminizip-dev \
  libopenal-dev libplib-dev libexpat1-dev libcjson-dev libutf8proc-dev libglm-dev

git -C "$SD" submodule update --init --recursive --depth 1

cmake -S "$SD" -B "$SD/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_LIBDIR=lib \
  -DOPTION_OSGGRAPH=ON -DOPTION_TRACKEDITOR=OFF -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build "$SD/build" -j"$(nproc)"
cmake --install "$SD/build"

# rlpilot native layer: worker host, robot module, client lib (installed into $PREFIX, next to the game)
cmake -S "$ROOT/native" -B "$ROOT/native/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/native/build" -j"$(nproc)"
cmake --install "$ROOT/native/build"

# Python env (gym wrapper, SB3, MLflow). Python 3.14 like the Mac's, so that checkpoints load on both.
command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/ then rerun"; exit 1; }
(cd "$ROOT" && uv sync --locked --extra train --extra dev --python 3.14)
