#!/usr/bin/env bash
# Build Speed Dreams on macOS (Apple Silicon, Homebrew). Run from rlpilot/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SD="$ROOT/speed-dreams-code"; DEPS="$ROOT/deps"; PREFIX="$ROOT/sd-install"

brew install cmake sdl2 sdl2_mixer sdl2_ttf open-scene-graph openal-soft enet expat jpeg \
  libpng curl rhash minizip cjson utf8proc freetype libvorbis libogg glm

git -C "$SD" submodule update --init --recursive --depth 1

# plib (not in Homebrew): Speed Dreams' CMake fork, static, minus the 'sl' sound lib (Carbon-only on macOS)
if [ ! -f "$DEPS/lib/libplibssg.a" ]; then
  T="$(mktemp -d)"; curl -sSL -o "$T/plib.tgz" https://files.speed-dreams.net/public/87a261d17faf7b88420ae92c1969553a
  echo "b659039748592399cdccda0204d5e4c495a6b3b30c8cd82b98b930ff02512b4a  $T/plib.tgz" | shasum -a 256 -c -
  tar xf "$T/plib.tgz" -C "$T"; P="$T/plib-v1.8.5-sd3"
  sed -i '' 's/!finite(a)/!isfinite(a)/' "$P/src/sg/sg.cxx"
  cmake -S "$P" -B "$P/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=98 -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  cmake --build "$P/build" -j --target plibul plibjs plibsg plibssg plibssgaux plibfnt
  mkdir -p "$DEPS/include/plib" "$DEPS/lib"
  cp "$P"/src/{util,js,sg,ssg,ssgAux,fnt,sl,pui,puAux,pw,net,psl}/*.h "$DEPS/include/plib/"
  cp "$P"/build/libplib{ul,js,ssg,ssgaux,fnt}.a "$DEPS/lib/"
  # static sg needs ul (Linux gets this for free from shared libs)
  libtool -static -o "$DEPS/lib/libplibsg.a" "$P/build/libplibsg.a" "$P/build/libplibul.a"
fi

cmake -S "$SD" -B "$SD/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH="$DEPS;/opt/homebrew;/opt/homebrew/opt/curl;/opt/homebrew/opt/jpeg" \
  -DOPENAL_INCLUDE_DIR=/opt/homebrew/opt/openal-soft/include \
  -DOPENAL_LIBRARY=/opt/homebrew/opt/openal-soft/lib/libopenal.dylib \
  -DOPTION_OSGGRAPH=ON -DOPTION_TRACKEDITOR=OFF -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build "$SD/build" -j
cmake --install "$SD/build"
cp "$ROOT/sd-local/config/raceman/rlbench.xml" "$PREFIX/share/games/speed-dreams-2/config/raceman/" 2>/dev/null || true
