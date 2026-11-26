# watch.py

Replays a checkpoint in the game, with live telemetry against the bot.

## Laps

```sh
uv run python scripts/watch.py sw30m-auto-both-pressure-gate05-cap05 ardennen-spa
uv run python scripts/watch.py sw30m-auto-both-pressure-gate05-cap05 torino --laps 1
uv run python scripts/watch.py sw30m-auto-both-pressure-gate05-cap05 ardennen-spa --best
uv run python scripts/watch.py runs/<run>/checkpoints/0014008320.pt ardennen-spa
uv run python scripts/watch.py sw30m-auto-both-pressure-gate05-cap05 ardennen-spa --no-game
uv run python scripts/watch.py sw30m-auto-both-pressure-gate05-cap05 ardennen-spa --shift sample
```

## Scenarios

```sh
uv run python scripts/watch.py sw30m-auto-both-pressure-gate05-cap05 --failed crashed
uv run python scripts/watch.py sw30m-auto-both-pressure-gate05-cap05 --failed all
uv run python scripts/watch.py sw30m-auto-both-pressure-gate05-cap05 --scenario Imula:700:crashed
uv run python scripts/watch.py runs/<run>/checkpoints/0014008320.pt --scenario Imula:700:crashed
```

Scenario ids are `track:metres:tier`. Tiers: `nominal`, `mild`, `perturbed`, `disturbed`, `crashed`.

## Flags

| flag | |
|---|---|
| `--laps N` | default 2 |
| `--best` | `best.pt` instead of the last checkpoint |
| `--seed N` | when the experiment has several seeds |
| `--no-game` | lap times + summary image only |
| `--scenario ID` | one suite scenario instead of laps |
| `--failed TIER` | list failed scenarios, then stop |
| `--shift argmax\|sample\|hazard` | how the gear choice is read |
| `--test` | allow test tracks |

## Output

`runs/<run>/watch/`: `<name>-<checkpoint>.png` (summary), `.npz` (per-step telemetry), `.controls` (the replay).

## Good ones to show

```sh
uv run python scripts/watch.py sw100m-auto-both-pressure-gate05-cap05-idle05-long alicante
uv run python scripts/watch.py sw100m-auto-both-pressure-gate05-cap05-idle05-long karwada
uv run python scripts/watch.py sw100m-auto-both-pressure-gate05-cap05-idle05-long ardennen-spa
uv run python scripts/watch.py runs/sw30m-auto-both-pressure-gate05-cap05__1__1795637869/checkpoints/0014008320.pt --scenario Imula:700:crashed
uv run python scripts/watch.py sw30m-auto-both-pressure-gate05-cap05 --scenario Imula:700:crashed
```
