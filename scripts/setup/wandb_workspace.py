"""The W&B workspace views of project RLPilot, kept in code: run it again after editing to update them in place.

    uv run python scripts/setup/wandb_workspace.py            # creates or updates both views, prints their links

- "Compare setups": runs grouped by --exp-name (the seeds of one setup), each chart one metric as the mean over seeds
  with a min-max band, most telling first.
- "One run": no grouping, related metrics overlaid on one chart; tick one run at a time.

The script owns the charts and the view settings. Which runs are shown, filters and colours are left as you set them
in the UI (kept from the saved view when it's updated).
"""

import argparse

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from wandb_workspaces._graphql import get_app_url
from wandb_workspaces.workspaces import internal

from rlpilot.env import ENDS
from rlpilot.evaluate import LOST_CAUSES
from rlpilot.starts import KINDS, SUITE_TIERS

PROJECT = "RLPilot"
X = "global_step"
RATE = (0, 1)  # y-range of every share, so charts read the same across views
VIEWS = {"compare": ("rlpilotcompare", "Compare setups"), "run": ("rlpilotrun", "One run")}  # key: (link id, title)


def ev(split, name):
    return f"3_eval/{split}/{name}"


def drv(name):
    return f"4_driving/{name}"


def ppo(name):
    return f"2_ppo/{name}"


# ---------------------------------------------------------------- compare setups: one metric per chart, mean + band


def chart(metric, title, rate=False):
    return wr.LinePlot(
        title=title, x=X, y=[metric], range_y=RATE if rate else (None, None),
        groupby_aggfunc="mean", groupby_rangefunc="minmax",
    )


def section(name, panels, columns, open_=True):
    rows = -(-len(panels) // columns)  # every chart on one page: no paging
    return ws.Section(
        name=name, panels=panels, is_open=open_, layout_settings=ws.SectionLayoutSettings(columns=columns, rows=rows)
    )


def tiers(split, outcome):
    return [chart(ev(split, f"{t}/{outcome}"), f"{t} {outcome}, {split}", rate=True) for t in SUITE_TIERS]


def compare_sections():
    v = "validation"
    return [
        section("1 Verdict: better, faster, still recovering", [
            chart(ev(v, "nominal/survival"), "nominal survival, validation (selects best.pt)", rate=True),
            chart(ev(v, "pace"), "pace, validation (1 = the bot)"),
            chart(ev(v, "crashed/completion"), "crashed completion, validation (recovery)", rate=True),
            chart(ev(v, "disturbed/survival"), "disturbed survival, validation", rate=True),
        ], columns=4),
        section("2 Generalization: train vs validation", [
            chart(ev("train", "nominal/survival"), "nominal survival, train", rate=True),
            chart(ev(v, "nominal/survival"), "nominal survival, validation", rate=True),
            chart(ev("train", "pace"), "pace, train"),
            chart(ev(v, "pace"), "pace, validation"),
        ], columns=4),
        section("3 Robustness by tier, validation", tiers(v, "survival") + tiers(v, "completion"), columns=5),
        section("4 How it fails", [
            *[chart(ev(v, f"nominal/lost_{c}"), f"nominal scenarios that lost control by {c}, validation", rate=True)
              for c in LOST_CAUSES],
            chart(drv("end/no_progress"), "training episodes ended: no progress (giving up)", rate=True),
            chart(drv("end/damage"), "training episodes ended: damage", rate=True),
            chart(drv("end/time_limit"), "training episodes ended: time limit", rate=True),
        ], columns=3),
        section("5 Driving, training episodes", [
            chart(drv("mean_speed"), "mean speed (m/s)"),
            chart(drv("offroad_frac"), "time off the track", rate=True),
            chart(drv("failed_near_start"), "failed near the start", rate=True),
            chart(drv("damage"), "damage per episode"),
            chart(drv("shifts_per_min"), "shifts per minute"),
            chart(drv("refused_shifts"), "refused shifts"),
        ], columns=3),
        section("6 PPO health", [
            chart(ppo("entropy"), "entropy"),
            chart(ppo("std_steer"), "action std, steer"),
            chart(ppo("std_steer_applied"), "action std, steer, after the speed scaling (rollout mean)"),
            chart(ppo("std_steer_slow"), "action std applied, steer, below 20 m/s"),
            chart(ppo("std_steer_fast"), "action std applied, steer, above 40 m/s"),
            chart(ppo("std_pressure_steer_slow"), "std pressure, steer, below 20 m/s (< 0: noise costs)"),
            chart(ppo("std_pressure_steer_fast"), "std pressure, steer, above 40 m/s (< 0: noise costs)"),
            chart(ppo("std_pedal"), "action std, pedal"),
            chart(ppo("explained_variance"), "explained variance (critic)"),
            chart(ppo("approx_kl"), "approx KL (update size)"),
            chart(ppo("clipfrac"), "clip fraction", rate=True),
        ], columns=3),
        section("7 PPO losses", [
            chart(ppo("value_loss"), "value loss"),
            chart(ppo("policy_loss"), "policy loss"),
            chart(ppo("learning_rate"), "learning rate"),
        ], columns=3, open_=False),
        section("8 Clean by tier, validation", tiers(v, "clean"), columns=5, open_=False),
        section("9 Tiers, train split",
                tiers("train", "survival") + tiers("train", "completion") + tiers("train", "clean"),
                columns=5, open_=False),
        section("10 Perf", [chart("5_perf/SPS", "env steps/s"), chart("5_perf/eval_seconds", "eval seconds")],
                columns=2, open_=False),
    ]


# ---------------------------------------------------------------- one run: related metrics overlaid


def overlay(metrics, title, rate=False, stacked=False):
    return wr.LinePlot(
        title=title, x=X, y=list(metrics), range_y=RATE if rate else (None, None),
        plot_type="stacked-area" if stacked else "line",
    )


def run_sections():
    v, t = "validation", "train"
    by_tier = lambda split: [  # noqa: E731
        overlay([ev(split, f"{tier}/{o}") for o in ("survival", "completion", "clean")], f"{tier}, {split}", rate=True)
        for tier in SUITE_TIERS
    ]
    return [
        section("1 Headline", [
            overlay([ev(v, f"nominal/{k}") for k in ("survival", "survival_lo", "survival_hi")],
                    "nominal survival, validation, with its 90% interval over tracks", rate=True),
            overlay([ev(t, f"nominal/{k}") for k in ("survival", "survival_lo", "survival_hi")],
                    "nominal survival, train, with its 90% interval over tracks", rate=True),
            overlay([ev(t, "pace"), ev(v, "pace")], "pace, train vs validation (1 = the bot)"),
            overlay([ev(v, "crashed/completion"), ev(v, "disturbed/survival")],
                    "recovery, validation: crashed completion, disturbed survival", rate=True),
        ], columns=4),
        section("2 Tiers, validation: survival, completion, clean", by_tier(v), columns=5),
        section("3 How it fails", [
            overlay([drv(f"end/{e}") for e in ENDS], "how training episodes end", rate=True, stacked=True),
            overlay([drv(f"lost_{c}") for c in LOST_CAUSES], "losses of control per training episode, by cause"),
            overlay([ev(v, f"nominal/lost_{c}") for c in LOST_CAUSES],
                    "nominal scenarios that lost control, validation, by cause", rate=True),
            overlay([drv(f"failed_near_start/{k}") for k in KINDS], "failed near the start, by kind of start",
                    rate=True),
        ], columns=4),
        section("4 Reward, training episodes", [
            overlay([drv("return")], "return"),
            overlay([drv(f"reward_{c}") for c in ("progress", "damage", "control", "terminal")],
                    "reward components per episode"),
        ], columns=2),
        section("5 Driving, training episodes", [
            overlay([drv("mean_speed")], "mean speed (m/s)"),
            overlay([drv("distance")], "distance (m)"),
            overlay([drv("seconds")], "episode length (s)"),
            overlay([drv("offroad_frac")], "time off the track", rate=True),
            overlay([drv("damage")], "damage per episode"),
            overlay([drv("shifts_per_min"), drv("refused_shifts")], "shifts per minute, refused shifts"),
        ], columns=3),
        section("6 PPO", [
            overlay([ppo("std_steer"), ppo("std_steer_applied"), ppo("std_pedal")], "action std, steer and pedal"),
            overlay([ppo("std_steer_slow"), ppo("std_steer_fast"), ppo("std_pedal_slow"), ppo("std_pedal_fast")],
                    "action std applied, below 20 and above 40 m/s"),
            overlay([ppo("std_pressure_steer_slow"), ppo("std_pressure_steer_fast"), ppo("std_pressure_pedal_slow"),
                     ppo("std_pressure_pedal_fast")], "std pressure (A x (z^2 - 1); < 0: noise costs)"),
            overlay([ppo("entropy")], "entropy"),
            overlay([ppo("explained_variance")], "explained variance (critic)"),
            overlay([ppo("approx_kl")], "approx KL"),
            overlay([ppo("clipfrac")], "clip fraction", rate=True),
            overlay([ppo("value_loss")], "value loss"),
            overlay([ppo("policy_loss")], "policy loss"),
            overlay([ppo("learning_rate")], "learning rate"),
        ], columns=4, open_=False),
        section("7 Tiers, train split", by_tier(t), columns=5, open_=False),
        section("8 Perf", [overlay(["5_perf/SPS"], "env steps/s"), overlay(["5_perf/eval_seconds"], "eval seconds")],
                columns=2, open_=False),
    ]


# ---------------------------------------------------------------- saving


def build(key, entity):
    link, title = VIEWS[key]
    name = internal._url_query_str_to_internal_name(link)  # W&B's internal view name: nw-<link id>-v
    w = ws.Workspace(
        entity=entity, project=PROJECT, name=title,
        sections=compare_sections() if key == "compare" else run_sections(),
        settings=ws.WorkspaceSettings(x_axis=X, smoothing_type="none", max_runs=100),
    )
    try:  # update in place: keep the runs shown, filters and colours set in the UI
        old = ws.Workspace._from_model(internal.View.from_name(entity, PROJECT, name))
        w.runset_settings = old.runset_settings
        w._internal_id = old._internal_id
    except ValueError:  # not created yet
        pass
    if key == "compare":
        w.runset_settings.groupby = [ws.Metric("Group")]
    else:
        w.runset_settings.groupby = []
    w._internal_name = name
    w.save()
    # the library's w.url drops the app's base path (forge.coreweave.com/wandb/)
    return f"{get_app_url(wandb.Api()).rstrip('/')}/{entity}/{PROJECT}?nw={link}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default=None, help="default: the logged-in account")
    p.add_argument("--only", choices=list(VIEWS), help="update only this view")
    a = p.parse_args()
    entity = a.entity or wandb.Api().default_entity
    for key in [a.only] if a.only else VIEWS:
        print(f"{VIEWS[key][1]}: {build(key, entity)}")


if __name__ == "__main__":
    main()
