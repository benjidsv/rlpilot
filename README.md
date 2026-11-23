# rlpilot (v2, work in progress)

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-VectorEnv-555555)
![C++](https://img.shields.io/badge/C++-17-00599C?logo=cplusplus&logoColor=white)

> **This branch is a work in progress.** It's a clean restart of the project. For the finished v1 (PPO over seven
> 300M-step runs, its results and findings), see the [`main` branch](https://github.com/benjidsv/rlpilot/tree/main).

A PPO driver for a racing simulator (Speed Dreams, a TORCS fork). The question is still generalisation: **can one policy
drive tracks it has never trained on?** The goal is safe before fast.

## Why a v2

v1 answered a lot of questions, but the code grew by accretion: about 8.7k lines, 25 env flags, SB3 subclasses, and an
evaluation too noisy to compare runs with. v2 keeps the fast C++ simulator wrapper and rebuilds the rest so that every
part is small and readable:

- **Training is a single file:** [CleanRL](https://github.com/vwxyzjn/cleanrl)'s `ppo_continuous_action.py`, with
  every change marked. One of them is a hybrid policy: a Gaussian over steering and pedal, and a discrete
  down / hold / up gear shift.
- **The car starts from odd situations, not just the grid:**
  - on the racing line at racing speed,
  - off-line, too fast or sideways,
  - mid-slide after a forced mistake,
  - or stopped and facing the wrong way after a crash.

  Training samples are spread per metre of road across 32 tracks.
- **Mistakes aren't the end:** an episode only fails after 10 s without progress, so the car can learn to recover,
  reverse included.
- **Evaluation is a fixed suite of 2,855 short scenarios:** start points every 200 m on the held-out tracks (every
  600 m on training tracks), with five levels of difficulty at each. Every checkpoint plays the same list, so results are reproducible, and two runs can be compared
  scenario by scenario.
- **The final exam is the Nordschleife:** 21 km, never seen in training, held out with two short tracks until the very
  end.

The Python package is about 1.2k lines. Design and decisions: [docs/v2.md](docs/v2.md).

## Status

- **Phase 1 is done:** the base, with a deliberately plain observation and reward. A 10M-step check run (one seed,
  20 min on an M1 Pro) survives 66% of nominal scenarios on the validation tracks, at about the built-in bot's pace
  where it survives. That's a sanity check, not a result.
- **Phase 2 is next:** designing the observation, the reward and the steering, one change at a time, judged on the
  suite.

## Quick start

Requires macOS 14.4+ on Apple Silicon with [Homebrew](https://brew.sh), or Linux (`build-linux.sh`), and
[uv](https://docs.astral.sh/uv/).

```sh
git clone --recurse-submodules -b v2 https://github.com/benjidsv/rlpilot.git
cd rlpilot
./build-macos.sh

# Initialize the game's settings dir once. The race name doesn't exist on purpose:
# the game populates the dir, prints "No such race type" and exits.
./sd-install/games/speed-dreams-2 -lc "$PWD/sd-local/" -s init -x

# Tracks and their bot references (41 tracks), then the tests and a run
uv run python scripts/download_tracks.py --categories circuit road gprix
uv run python scripts/setup_tracks.py
uv run pytest
uv run wandb login   # once; metrics go to W&B, project RLPilot
uv run python scripts/train.py --exp-name first --total-timesteps 50000000
```

## Repository

| Path | What it is |
|---|---|
| `python/rlpilot/` | Env (`env.py`, `starts.py`, `controls.py`, `obs.py`, `reward.py`), tracks (`tracks.py`), scenario suite (`evaluate.py`), simulator client (`client.py`) |
| `scripts/` | `train.py` (CleanRL PPO), `eval.py`, `compare.py`, split and suite generation, track setup |
| `data/` | Race configs, the frozen track split (`split-v2.json`) and scenario suite (`suite-v2.csv`) |
| `native/` | C++ worker host, robot shim, shared-memory protocol, client library, C benchmark |
| `speed-dreams-code/` | Submodule: [my fork](https://github.com/benjidsv/speed-dreams-code) of [Speed Dreams](https://forge.a-lec.org/speed-dreams/speed-dreams-code), upstream plus build fixes and launch hooks |

The simulator wrapper and much of v2's code were written with AI assistance (Claude Code), from my requirements and
design decisions. I don't send changes upstream: Speed Dreams' `NOAI.md` policy rejects AI-assisted contributions.

## License

[GPL-2.0-or-later](LICENSE), the same as Speed Dreams. Tracks, cars and other game assets come from the Speed Dreams
project and keep their own licences; this repo doesn't include them.
