"""Sanity tests. Run:  python3 test_fearless_engine.py   (or: pytest test_fearless_engine.py)"""
import numpy as np
from fearless_engine import (FearlessEngine, GameConfig, MatchupModel, GENERIC_POLICY,
                             RANDOM_BAN_POLICY, WASTED_BAN_POLICY)

rng_s = np.random.default_rng(0)
STRENGTHS = rng_s.normal(0, 0.05, 24)


def eng(best_of=3, strengths=STRENGTHS, **kw):
    return FearlessEngine(strengths, config=GameConfig(best_of=best_of, **kw))


def test_equal_champions_give_half():
    for bo in (1, 3, 5):
        e = eng(bo, strengths=np.zeros(24))
        assert abs(e.series_win_prob(GENERIC_POLICY, GENERIC_POLICY, "A") - 0.5) < 1e-12


def test_closed_form_series_probability():
    d = 0.4
    p = 1 / (1 + np.exp(-d))
    expected = {1: p, 3: p**2 * (3 - 2 * p), 5: p**3 * (10 - 15 * p + 6 * p**2)}
    for bo, exp in expected.items():
        e = eng(bo, strengths=np.zeros(24), delta=d)
        assert abs(e.series_win_prob(GENERIC_POLICY, GENERIC_POLICY, "A") - exp) < 1e-12


def test_side_symmetry_for_identical_teams():
    for bo in (3, 5):
        e = eng(bo)
        a = e.series_win_prob(GENERIC_POLICY, GENERIC_POLICY, "A")
        b = e.series_win_prob(GENERIC_POLICY, GENERIC_POLICY, "B")
        assert abs(a + b - 1) < 1e-12


def test_fearless_lock_both_teams():
    e = eng(5, lock_rule="both")
    rng = np.random.default_rng(1)
    for _ in range(200):
        res = e.simulate_series(RANDOM_BAN_POLICY, RANDOM_BAN_POLICY, rng)
        used = []
        for g in res.games:
            assert len(set(g.bans.values()) | set(g.picks.values())) == 4   # distinct in a game
            used += list(g.picks.values())
        assert len(used) == len(set(used))                                  # never re-picked


def test_fearless_lock_own_only():
    e = eng(5, lock_rule="own")
    rng = np.random.default_rng(2)
    for _ in range(200):
        res = e.simulate_series(RANDOM_BAN_POLICY, RANDOM_BAN_POLICY, rng)
        for team in "AB":
            mine = [g.picks[team] for g in res.games]
            assert len(mine) == len(set(mine))


def test_bans_never_target_locked_champions():
    e = eng(5)
    rng = np.random.default_rng(3)
    for _ in range(200):
        res = e.simulate_series(RANDOM_BAN_POLICY, RANDOM_BAN_POLICY, rng)
        locked = set()
        for g in res.games:
            assert not (set(g.bans.values()) & locked)
            locked |= set(g.picks.values())


def test_monte_carlo_matches_exact():
    e = eng(3)
    exact = e.series_win_prob(GENERIC_POLICY, WASTED_BAN_POLICY, "both")
    mc, se = e.monte_carlo(GENERIC_POLICY, WASTED_BAN_POLICY, n=20000, seed=5)
    assert abs(mc - exact) < 4 * se + 1e-3


def test_matchup_matrix_must_be_antisymmetric():
    try:
        MatchupModel(3, np.ones((3, 3)))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
