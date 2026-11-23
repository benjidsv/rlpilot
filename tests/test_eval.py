"""The scenario suite: reproducible results, metrics, the paired comparison."""
import sys
from pathlib import Path

from conftest import scripted
from rlpilot import evaluate

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from compare import sign_test  # noqa: E402


def small_suite():
    suite = evaluate.load_suite(("validation",))
    keep = {t: sorted({r["s"] for r in suite if r["track"] == t})[:3] for t in ("dijon", "street-1")}
    return [r for r in suite if r["track"] in keep and r["s"] in keep[r["track"]]]


def test_suite_is_reproducible():
    suite = small_suite()
    assert len(suite) == 2 * 3 * 5
    a = evaluate.run_suite(scripted, suite, n_workers=6)
    b = evaluate.run_suite(scripted, suite, n_workers=6)
    assert a == b
    assert [r["id"] for r in a] == [r["id"] for r in suite]
    # the layout doesn't matter: every scenario is a fresh race, its launch overlapped with the others' steps or not
    assert evaluate.run_suite(scripted, suite, n_workers=2) == a
    s = evaluate.Suite(suite, n_workers=6)  # nor do earlier runs on the same workers (train.py keeps them up)
    try:
        assert s.run(scripted) == a and s.run(scripted) == a
    finally:
        s.close()
    m = evaluate.summarize(a)
    assert 0 <= m["validation/nominal/survival"] <= 1 and "validation/crashed/completion" in m


def test_sign_test():
    assert sign_test(0, 0) == 1.0
    assert sign_test(10, 10) == 1.0
    assert sign_test(20, 2) < 0.001
