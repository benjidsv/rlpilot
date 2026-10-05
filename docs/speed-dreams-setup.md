# Speed Dreams on macOS — setup notes

Native build of [Speed Dreams](https://forge.a-lec.org/speed-dreams/speed-dreams-code) (TORCS fork) on Apple Silicon, as the simulator behind a future gym RL environment. Tested 2026-10-05 on an M1 Pro, macOS 26.5, Homebrew, CMake 4.2.

## Layout

```
rlpilot/
├── speed-dreams-code/      submodule: my fork, branch `rlpilot` = upstream 62c97b7c6 + macOS fixes
│   ├── speed-dreams-data/  submodule: base assets (1 track: Jarama, 1 car: sc-cavallo-360)
│   ├── freesolid/          submodule: collision lib, built in-tree
│   └── build/              CMake build dir
├── deps/                   plib (static), built from source; not in Homebrew
├── sd-install/             `cmake --install` prefix: binary, modules, data
├── sd-local/               game's user/settings dir, passed with `-lc`; downloads land here too
└── build-macos.sh          full reproducible build: brew deps → plib → configure → build → install
```

## Build

```sh
./build-macos.sh
```

Things that differ from the Linux README. The "(patch)" fixes are committed on the fork's `rlpilot` branch:

| Problem | Fix |
|---|---|
| `plib` missing from Homebrew | Build SD's CMake fork `plib-v1.8.5-sd3` (URL and SHA-256 in the script). Patch `finite`→`isfinite`. Skip `sl`: its macOS backend uses the removed Carbon Sound Manager. |
| Static `libplibsg` needs `libplibul` | Merge them with `libtool -static` (on Linux, shared libs hide this) |
| Root CMakeLists sets `CMAKE_FIND_LIBRARY_SUFFIXES .so` on all UNIX | Use `.dylib .tbd .a` on APPLE (patch) |
| C++11 forced on APPLE | Raised to 17 (patch) |
| Apple OpenAL framework uses `<OpenAL/al.h>` | Point `OPENAL_*` at keg-only `openal-soft` |
| `snddefault` compiles plib sound | Excluded on APPLE; OpenAL only (patch) |
| `networkhuman` misses the enet include dir; `grSky.cpp` misses `<cmath>` | One-line patches |
| SDL asks for a GL 3.2 **Core** context on macOS, but the renderers are fixed-function | Request a 2.1 legacy context like Linux (`tgfclient/guiscreen.cpp`, patch) |

Configure notes:
- `OPTION_USE_MACPORTS` gets forced on for APPLE. That's harmless: `CMAKE_PREFIX_PATH` overrides it.
- `OPTION_OSGGRAPH=ON` (the default renderer) is set explicitly.
- `OPTION_TRACKEDITOR=OFF`, because the track editor needs Java.

## Run

Always pass `-lc <abs path>/` so nothing goes to `~/.speed-dreams-2`. The trailing slash is required.

```sh
SD=./sd-install/games/speed-dreams-2; LC=$PWD/sd-local/
$SD -lc $LC                         # menus (Race → mode → Players → Garage to pick a car)
$SD -lc $LC -s rlbench -x           # headless race, AI bot (simplix)
$SD -lc $LC -s rlbench              # same race, rendered
$SD -lc $LC -s humantest            # you drive: mouse steer, ↑/↓ throttle/brake
```

Flags: `-tl <0-5>` sets log verbosity, `-nr` makes randomness reproducible.

### Config gotchas
- **Race configs** (`config/raceman/<name>.xml`) must exist in **both** `sd-install/share/games/speed-dreams-2/config/raceman/` and `sd-local/config/raceman/`. The race list is enumerated from the install dir, then loaded from the local dir. Otherwise you get `No such race type`.
- **The engine rewrites race configs when it loads them.** It drops the driver `idx` and resets `display mode` (headless runs set it to `results only`, so rendered runs need `normal`). Keep master copies elsewhere and regenerate them before each run.
- **AI bot instances** are defined as `sd-local/drivers/<robot>/<N>/driver.xml` (plus `skill.xml`). The engine regenerates `drivers/<robot>/<robot>.xml` from these at startup, so editing that file directly gets overwritten.
- In a race config, a driver is `module` (`simplix`, `human`, …) plus `idx` (the instance number).
- The first run with a new `-lc` dir populates it from defaults and overwrites anything already in it.

## Performance (headless, one car, simplix bot, Jarama)

| | Sim time / wall time |
|---|---|
| 1 process (30 laps, 3,220 s simulated in 2.6 s) | **~1,300× real time** |
| 8 processes in parallel | **~6,000× real time** in aggregate |

- Physics runs at 500 Hz (`RCM_MAX_DT_SIMU`). Robots are called at 50 Hz (`RCM_MAX_DT_ROBOTS`), which is about 65k agent steps/s per process.
- Runs are deterministic: identical runs give identical race times.
- Process startup costs about 0.3 s headless and about 2 s rendered.

## Pointers for the gym wrapper
Implemented: see [gym-wrapper.md](gym-wrapper.md). The notes below are the original research.

- **Agent hook:** a robot module, i.e. a `.so` in `src/drivers/<name>/` built with the `ROBOT_MODULE` CMake macro. `simplix`/`axiom` are complete examples. `drive(tCarElt*, tSituation*)` runs every 20 ms and sets `car->_steerCmd`, `_accelCmd`, `_brakeCmd`, `_gearCmd`.
- **Observations:** car state is in `tCarElt` (`src/interfaces/car.h`); track geometry is `tTrackSeg` (`track.h`).
- **Headless UI:** `src/modules/userinterface/textonly/`.
- **Expected bottlenecks:** passing data to Python every step, and resets (relaunching the process). The simulator itself won't be the limit.
- **Licence:** upstream is GPLv2+. `NOAI.md` rejects AI-assisted contributions, so keep these changes local and don't upstream them.

## Git

Three repos are involved:
- **`rlpilot`** (`github.com/benjidsv/rlpilot`): my own work. It contains the wrapper, docs, build script and master configs.
- **`speed-dreams-code`**: a submodule pointing to my fork (`github.com/benjidsv/speed-dreams-code`), branch `rlpilot`. Remotes: `origin` = fork, `upstream` = `forge.a-lec.org/speed-dreams/speed-dreams-code`.
- **`speed-dreams-data` and `freesolid`**: submodules of the engine, pinned to upstream. I never edit them.

Ignored in `rlpilot`: `deps/`, `sd-install/`, `sd-local/`, `*.log`. The engine's `build/` is ignored by upstream's own `.gitignore`.

### Day to day

**Change the engine.** Commit in the submodule first, then record the new pointer in `rlpilot`:
```sh
cd speed-dreams-code
git switch rlpilot              # submodules start on a detached HEAD; check out the branch before committing
git commit -am "…" && git push
cd ..
git add speed-dreams-code && git commit -m "Bump speed-dreams-code"
```

**Pull in upstream updates.**
```sh
cd speed-dreams-code
git fetch upstream
git rebase upstream/main
git push --force-with-lease
git submodule update --init --recursive     # upstream may have moved the data/freesolid pins
cd .. && git add speed-dreams-code && git commit -m "Rebase engine on upstream"
```
Then rebuild with `./build-macos.sh`.

**Fresh clone.**
```sh
git clone --recurse-submodules https://github.com/benjidsv/rlpilot.git
cd rlpilot && ./build-macos.sh
```

**After pulling `rlpilot` on another machine:** run `git submodule update --init --recursive`.

Reminders:
- Remotes use HTTPS (auth via `gh auth login` + `gh auth setup-git`), not SSH.
- Never open merge requests upstream from the fork (see `NOAI.md`).
