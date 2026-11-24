"""Sanity tests. Run:  python3 test_fearless_engine.py   (or: pytest test_fearless_engine.py)"""
import os
import numpy as np
import pandas as pd
from fearless_engine import (FearlessEngine, GameConfig, MatchupModel, GENERIC_POLICY,
                             RANDOM_BAN_POLICY, WASTED_BAN_POLICY, ProficiencyParams, proficiency_value, MatchupData,
                             ProficiencyModel, TARGETED_POLICY, targeted_policy)

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


def test_never_played_is_neutral():
    assert proficiency_value([np.nan, 0], [np.nan, 0.9]).tolist() == [0.0, 0.0]


def test_higher_win_rate_more_proficient():
    wr = np.linspace(0, 1, 11)
    for g in (1, 5, 20, 36):
        pi = proficiency_value(np.full(11, g), wr)
        assert np.all(np.diff(pi) > 0)


def test_more_games_more_proficient_when_win_rate_at_least_50():
    g = np.arange(1, 37)
    for w in (0.5, 0.6, 0.8, 1.0):
        pi = proficiency_value(g, np.full(36, w))
        assert np.all(np.diff(pi) >= 0), w


def test_small_sample_is_shrunk_and_bounded():
    p = ProficiencyParams()
    one_game_perfect = proficiency_value([1], [1.0])[0]
    twenty_games_70 = proficiency_value([20], [0.7])[0]
    assert one_game_perfect < twenty_games_70
    assert np.all(np.abs(proficiency_value([1, 36, 50], [0.0, 1.0, 1.0])) <= p.cap)


def test_unknown_win_rate_still_rewards_experience():
    assert proficiency_value([10], [np.nan])[0] > 0


def test_real_workbook_integration():
    csv = os.environ.get("FEARLESS_CSV", "All_Top_Lane_Champions_Worlds_2026.csv")
    xlsx = os.environ.get("FEARLESS_XLSX", "2026_worlds_top_laner_champion_win_rates.xlsx")
    if not (os.path.exists(csv) and os.path.exists(xlsx)):
        print("  (skipped: data files not found; set FEARLESS_CSV / FEARLESS_XLSX)")
        return
    e = FearlessEngine.from_csv(csv, player_xlsx=xlsx)
    pp = e.players
    assert pp.table.shape == (16, 24) and not pp.table.isna().any().any()
    played = pp.games.fillna(0) > 0
    assert (pp.table.where(~played).fillna(0) == 0).all().all()       # blanks are exactly neutral
    e.set_players(pp.players[0], pp.players[1])
    assert np.all((e.P_A > 0) & (e.P_A < 1))
    assert not np.allclose(e.P_A, 1 / (1 + np.exp(-(e.strength[:, None] - e.strength[None, :]))))


def _table_from_logits(L, names):
    W = 1 / (1 + np.exp(-L)); np.fill_diagonal(W, np.nan)
    return pd.DataFrame(W, index=names, columns=names)


def test_pure_strength_table_has_no_matchup_effect():
    names = list("ABCDE"); u = np.array([0.3, 0.1, 0.0, -0.1, -0.3])
    md = MatchupData(_table_from_logits(u[:, None] - u[None, :], names), names)
    assert np.allclose(md.m, 0, atol=1e-9) and np.allclose(md.u - md.u.mean(), u - u.mean())


def test_counter_cycle_gives_signed_antisymmetric_edges():
    names = list("ABC"); L = np.zeros((3, 3))
    for a, b in ((0, 1), (1, 2), (2, 0)):          # rock-paper-scissors: A>B, B>C, C>A
        L[a, b], L[b, a] = 0.2, -0.2
    md = MatchupData(_table_from_logits(L, names), names, weight=1.0, cap=1.0)
    assert md.m[0, 1] > 0 and md.m[1, 2] > 0 and md.m[2, 0] > 0
    assert np.allclose(md.m, -md.m.T) and np.allclose(md.m.sum(axis=1), 0)


def test_matchup_weight_and_cap_bound_the_term():
    names = list("ABC"); L = np.array([[0, 2.0, -2.0], [-2.0, 0, 2.0], [2.0, -2.0, 0]])
    md = MatchupData(_table_from_logits(L, names), names, weight=0.5, cap=0.3)
    assert np.abs(md.m).max() <= 0.5 * 0.3 + 1e-12


def test_asymmetric_table_is_symmetrised():
    names = list("AB"); W = pd.DataFrame([[np.nan, 0.60], [0.45, np.nan]], index=names, columns=names)
    md = MatchupData(W, names)
    assert abs(md.max_asymmetry - 0.05) < 1e-9 and np.allclose(md.m, -md.m.T)


def test_real_matchup_workbook_integration():
    csv = os.environ.get("FEARLESS_CSV", "All_Top_Lane_Champions_Worlds_2026.csv")
    xlsx = os.environ.get("FEARLESS_MATCHUPS", "Champion_Matchup_Win_Rates.xlsx")
    if not (os.path.exists(csv) and os.path.exists(xlsx)):
        print("  (skipped: data files not found; set FEARLESS_CSV / FEARLESS_MATCHUPS)")
        return
    e = FearlessEngine.from_csv(csv, matchup_xlsx=xlsx)
    m = e.matchups.m
    assert m.shape == (24, 24) and np.allclose(m, -m.T) and np.abs(m).max() > 0
    assert np.allclose(e.P_A + e.P_A.T, 1)             # identical teams: P(a beats b) + P(b beats a) = 1
    assert np.all((e.P_A > 0) & (e.P_A < 1))


def _comfort_engine(lock_rule="both", best_of=1, n=8):
    s = np.zeros(n); s[0] = 0.5                          # champion 0 is the generically strongest
    prof = ProficiencyModel(n); prof.pi["B"][3] = 2.0    # team B's comfort champion is 3 (bonus beats the strength gap)
    return FearlessEngine(s, config=GameConfig(best_of=best_of, lock_rule=lock_rule), proficiency=prof)


def test_targeted_ban_hits_opponent_comfort_champion_generic_does_not():
    from fearless_engine import View
    e = _comfort_engine()
    legal = e.legal_bans(e.initial_state(), ())
    v = View("A", "ban", 1, "blue", legal, (), None, e.initial_state(), None)
    assert TARGETED_POLICY.ban_fn(e, v) == 3
    assert GENERIC_POLICY.ban_fn(e, v) == 0


def test_targeted_ban_beats_a_wasted_ban_against_a_comfort_champion():
    e = _comfort_engine()
    targeted = e.series_win_prob(TARGETED_POLICY, WASTED_BAN_POLICY, "A")
    wasted = e.series_win_prob(WASTED_BAN_POLICY, WASTED_BAN_POLICY, "A")
    assert targeted > wasted + 0.05


def test_targeted_ban_is_legal_in_every_state_under_both_lock_rules():
    for rule in ("both", "own"):
        e = _comfort_engine(rule, best_of=5, n=24)       # a Bo5 with locks needs a big enough pool
        e.series_win_prob(TARGETED_POLICY, targeted_policy(1.0), "both")      # raises if illegal
        rng = np.random.default_rng(4)
        for _ in range(100):
            e.simulate_series(TARGETED_POLICY, TARGETED_POLICY, rng)


def test_targeted_ban_is_deterministic_and_flagged_so():
    assert TARGETED_POLICY.deterministic


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
