#!/usr/bin/env python3
"""
fearless_engine.py -- Top-lane Fearless Draft game engine (Bo1 / Bo3 / Bo5).

Game structure (simplified top-lane draft)
------------------------------------------
Per game:  blue bans 1 -> red bans 1 -> blue picks (first pick) -> red picks (counterpick).
A banned or picked champion is unavailable to BOTH teams for the rest of that game.
Sides alternate each game.  Which team starts on blue is a parameter ("A", "B" or "both").

Fearless lock (config.lock_rule)
--------------------------------
"both" (default): every champion picked in a game is locked for BOTH teams in later games.
"own"           : a team may not re-pick its own earlier picks (kept for sensitivity analysis).
Banning a champion that is already locked for everyone is a wasted ban, so it is not allowed.

Game win probability (team A's point of view; log-odds)
-------------------------------------------------------
logit P(A wins) = s[a] - s[b] + m[a, b] + pi_A[a] - pi_B[b] + delta
    s      generic champion strength (from the CSV: win rate, pick rate, ban rate)
    m      matchup term, antisymmetric (m[a,b] = -m[b,a])        -> PLACEHOLDER (zeros)
    pi_T   team T's proficiency on a champion                    -> PLACEHOLDER (zeros)
    delta  team A's edge from the other four roles               -> 0 by default

Policies are objects with a ban function and a pick function, so strategies can be swapped
in and compared.  Deterministic policies can be evaluated EXACTLY (no sampling) with
FearlessEngine.series_win_prob(); any policy can be simulated with simulate_series().
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TEAMS = ("A", "B")


def other(team: str) -> str:
    return "B" if team == "A" else "A"


# --------------------------------------------------------------------------- #
# 1. Champion data and generic strength
# --------------------------------------------------------------------------- #
def _pct(x) -> float:
    return float(str(x).strip().rstrip("%"))


def load_champions(path: str, shrink_k: float = 2.0, pop_weight: float = 0.015) -> pd.DataFrame:
    """Read the champion CSV and add a log-odds 'strength' column.

    strength = w * logit(win rate)  +  pop_weight * z(pick rate + ban rate),   then centred.

    * w = pick / (pick + shrink_k) shrinks the win rate of rarely picked champions toward
      50% (pick rate is the only sample-size proxy in the file).  shrink_k is in
      percentage points of pick rate.
    * The popularity term gives champions that are picked or banned more often slightly
      more strength.  pop_weight is the log-odds bonus per standard deviation of
      (pick + ban); it is deliberately small and is a modelling assumption, not a finding.
    Only differences in strength enter the game model, so centring changes nothing.
    """
    df = pd.read_csv(path)
    for col in ("Win_Rate", "Pick_Rate", "Ban_Rate"):
        df[col] = df[col].map(_pct)
    df = df.dropna().reset_index(drop=True)

    wr = df["Win_Rate"] / 100.0
    df["wr_logit"] = np.log(wr / (1.0 - wr))
    df["shrink_w"] = df["Pick_Rate"] / (df["Pick_Rate"] + shrink_k)
    df["wr_component"] = df["shrink_w"] * df["wr_logit"]

    presence = df["Pick_Rate"] + df["Ban_Rate"]
    z = (presence - presence.mean()) / presence.std(ddof=0)
    df["pop_component"] = pop_weight * z

    s = df["wr_component"] + df["pop_component"]
    df["strength"] = s - s.mean()
    return df


# --------------------------------------------------------------------------- #
# 2. Placeholders for matchup and proficiency data
# --------------------------------------------------------------------------- #
class MatchupModel:
    """PLACEHOLDER. m[i, j] = log-odds edge for the team on champion i vs champion j.
    Zeros for now.  Supply an antisymmetric matrix (champion order = engine order)."""

    def __init__(self, n: int, matrix: Optional[np.ndarray] = None):
        m = np.zeros((n, n)) if matrix is None else np.asarray(matrix, dtype=float)
        if m.shape != (n, n):
            raise ValueError(f"matchup matrix must be {n}x{n}, got {m.shape}")
        if not np.allclose(m, -m.T):
            raise ValueError("matchup matrix must be antisymmetric: m[i,j] = -m[j,i]")
        self.m = m


class ProficiencyModel:
    """PLACEHOLDER. pi[team][c] = log-odds bonus for that team's player on champion c.
    Zeros for now.  Use set() when player proficiency data is available."""

    def __init__(self, n: int):
        self.pi: Dict[str, np.ndarray] = {t: np.zeros(n) for t in TEAMS}

    def set(self, team: str, vector) -> None:
        v = np.asarray(vector, dtype=float)
        if v.shape != self.pi[team].shape:
            raise ValueError("proficiency vector has the wrong length")
        self.pi[team] = v


# --------------------------------------------------------------------------- #
# 3. Configuration, state and records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GameConfig:
    best_of: int = 3                 # 1, 3 or 5
    lock_rule: str = "both"          # "both" | "own"
    delta: float = 0.0               # team A's rest-of-team edge (log-odds)

    def __post_init__(self):
        if self.best_of < 1 or self.best_of % 2 == 0:
            raise ValueError("best_of must be a positive odd number")
        if self.lock_rule not in ("both", "own"):
            raise ValueError("lock_rule must be 'both' or 'own'")

    @property
    def wins_needed(self) -> int:
        return (self.best_of + 1) // 2


@dataclass(frozen=True)
class SeriesState:
    game_no: int                     # number of the NEXT game (1-based)
    wins_A: int
    wins_B: int
    locked_A: frozenset
    locked_B: frozenset


@dataclass
class View:
    """Everything a policy may look at when it must act."""
    team: str
    phase: str                       # "ban" or "pick"
    game_no: int
    side: str                        # "blue" or "red"
    legal: np.ndarray                # legal champion indices
    banned: Tuple[int, ...]          # bans made so far this game
    opp_pick: Optional[int]          # opponent's pick this game, if already made
    state: SeriesState
    rng: Optional[np.random.Generator]


@dataclass
class DraftResult:
    game_no: int
    blue: str
    bans: Dict[str, int]
    picks: Dict[str, int]
    p_A: float                       # P(A wins this game)


@dataclass
class SeriesResult:
    winner: str
    wins_A: int
    wins_B: int
    games: List[DraftResult]


# --------------------------------------------------------------------------- #
# 4. Policies
# --------------------------------------------------------------------------- #
PolicyFn = Callable[["FearlessEngine", View], int]


@dataclass
class Policy:
    name: str
    ban_fn: PolicyFn
    pick_fn: PolicyFn
    deterministic: bool = True


def generic_ban(engine, view):
    """Ban the legal champion with the highest generic strength."""
    return int(view.legal[np.argmax(engine.strength[view.legal])])


def lowest_strength_ban(engine, view):
    """Baseline: effectively a wasted ban (weakest legal champion)."""
    return int(view.legal[np.argmin(engine.strength[view.legal])])


def random_choice(engine, view):
    return int(view.rng.choice(view.legal))


def best_response_pick(engine, view):
    """Counterpick: best response to the opponent's pick.
    First pick: maximin over the opponent's possible replies (one-game lookahead only)."""
    mine = engine.prob_matrix(view.team)             # mine[own champ, opponent champ]
    legal = view.legal
    if view.opp_pick is not None:
        return int(legal[np.argmax(mine[legal, view.opp_pick])])
    opp = other(view.team)
    opp_legal = engine.legal_picks(view.state, opp, view.banned, ())
    best, best_val = int(legal[0]), -1.0
    for c in legal:
        replies = opp_legal[opp_legal != c]
        val = mine[c, replies].min() if len(replies) else 1.0
        if val > best_val + 1e-12:
            best, best_val = int(c), val
    return best


GENERIC_POLICY = Policy("generic-ban + best-response pick", generic_ban, best_response_pick)
WASTED_BAN_POLICY = Policy("wasted-ban + best-response pick", lowest_strength_ban, best_response_pick)
RANDOM_BAN_POLICY = Policy("random-ban + best-response pick", random_choice, best_response_pick,
                           deterministic=False)


# --------------------------------------------------------------------------- #
# 5. The engine
# --------------------------------------------------------------------------- #
class FearlessEngine:
    def __init__(self, strengths, names=None, config: Optional[GameConfig] = None,
                 matchups: Optional[MatchupModel] = None,
                 proficiency: Optional[ProficiencyModel] = None):
        self.strength = np.asarray(strengths, dtype=float)
        self.n = len(self.strength)
        self.names = list(names) if names is not None else [f"champ{i}" for i in range(self.n)]
        self.cfg = config or GameConfig()
        self.matchups = matchups or MatchupModel(self.n)
        self.proficiency = proficiency or ProficiencyModel(self.n)
        self.refresh()

    @classmethod
    def from_csv(cls, path: str, config: Optional[GameConfig] = None,
                 shrink_k: float = 2.0, pop_weight: float = 0.015, **kw):
        df = load_champions(path, shrink_k, pop_weight)
        eng = cls(df["strength"].to_numpy(), df["Champion_Name"].tolist(), config, **kw)
        eng.table = df
        return eng

    def refresh(self) -> None:
        """Recompute win probabilities (call after changing matchup/proficiency/delta)."""
        s = self.strength
        logit = (s[:, None] - s[None, :] + self.matchups.m
                 + self.proficiency.pi["A"][:, None] - self.proficiency.pi["B"][None, :]
                 + self.cfg.delta)
        self.P_A = 1.0 / (1.0 + np.exp(-logit))                  # P_A[a, b]
        self._P = {"A": self.P_A, "B": 1.0 - self.P_A.T}        # [own, opponent]

    def prob_matrix(self, team: str) -> np.ndarray:
        return self._P[team]

    # ---- state ----------------------------------------------------------- #
    def initial_state(self) -> SeriesState:
        return SeriesState(1, 0, 0, frozenset(), frozenset())

    def is_over(self, state: SeriesState) -> bool:
        return max(state.wins_A, state.wins_B) >= self.cfg.wins_needed

    @staticmethod
    def blue_team(game_no: int, start_blue: str) -> str:
        return start_blue if game_no % 2 == 1 else other(start_blue)

    # ---- legality -------------------------------------------------------- #
    @staticmethod
    def _mask(n: int, *removed) -> np.ndarray:
        mask = np.ones(n, dtype=bool)
        for group in removed:
            for c in group:
                mask[c] = False
        return mask

    def legal_bans(self, state: SeriesState, already) -> np.ndarray:
        locked_for_all = state.locked_A & state.locked_B
        return np.flatnonzero(self._mask(self.n, locked_for_all, already))

    def legal_picks(self, state: SeriesState, team: str, banned, taken) -> np.ndarray:
        locked = state.locked_A if team == "A" else state.locked_B
        return np.flatnonzero(self._mask(self.n, locked, banned, taken))

    # ---- one game's draft ------------------------------------------------- #
    def play_draft(self, state: SeriesState, policies: Dict[str, Policy],
                   start_blue: str = "A", rng: Optional[np.random.Generator] = None) -> DraftResult:
        g = state.game_no
        blue = self.blue_team(g, start_blue)
        red = other(blue)
        side = {blue: "blue", red: "red"}

        def act(team, phase, legal, banned, opp_pick):
            if len(legal) == 0:
                raise RuntimeError(f"no legal {phase} for team {team} in game {g}; pool too small")
            view = View(team, phase, g, side[team], legal, tuple(banned), opp_pick, state, rng)
            pol = policies[team]
            c = int((pol.ban_fn if phase == "ban" else pol.pick_fn)(self, view))
            if c not in legal:
                raise ValueError(f"policy '{pol.name}' chose illegal {phase} {c}")
            return c

        bans: Dict[str, int] = {}
        for team in (blue, red):
            bans[team] = act(team, "ban", self.legal_bans(state, bans.values()), bans.values(), None)

        picks: Dict[str, int] = {}
        picks[blue] = act(blue, "pick", self.legal_picks(state, blue, bans.values(), ()),
                          bans.values(), None)
        picks[red] = act(red, "pick", self.legal_picks(state, red, bans.values(), (picks[blue],)),
                         bans.values(), picks[blue])

        return DraftResult(g, blue, bans, picks, float(self.P_A[picks["A"], picks["B"]]))

    def next_state(self, state: SeriesState, draft: DraftResult, a_won: bool) -> SeriesState:
        a, b = draft.picks["A"], draft.picks["B"]
        if self.cfg.lock_rule == "both":
            la, lb = state.locked_A | {a, b}, state.locked_B | {a, b}
        else:
            la, lb = state.locked_A | {a}, state.locked_B | {b}
        return SeriesState(state.game_no + 1, state.wins_A + int(a_won),
                           state.wins_B + int(not a_won), frozenset(la), frozenset(lb))

    # ---- simulation ------------------------------------------------------- #
    def simulate_series(self, pol_A: Policy, pol_B: Policy, rng: np.random.Generator,
                        start_blue: str = "A") -> SeriesResult:
        policies = {"A": pol_A, "B": pol_B}
        state, games = self.initial_state(), []
        while not self.is_over(state):
            draft = self.play_draft(state, policies, start_blue, rng)
            a_won = bool(rng.random() < draft.p_A)
            games.append(draft)
            state = self.next_state(state, draft, a_won)
        return SeriesResult("A" if state.wins_A > state.wins_B else "B",
                            state.wins_A, state.wins_B, games)

    def monte_carlo(self, pol_A: Policy, pol_B: Policy, n: int = 10000, seed: int = 0,
                    start_blue: str = "both") -> Tuple[float, float]:
        """Estimated P(A wins series) and its standard error (start_blue 'both' alternates)."""
        rng = np.random.default_rng(seed)
        starts = ["A", "B"] if start_blue == "both" else [start_blue]
        wins = [self.simulate_series(pol_A, pol_B, rng, starts[i % len(starts)]).winner == "A"
                for i in range(n)]
        p = float(np.mean(wins))
        return p, float(np.sqrt(p * (1 - p) / n))

    # ---- exact evaluation (deterministic policies) ------------------------- #
    def series_win_prob(self, pol_A: Policy, pol_B: Policy, start_blue: str = "A") -> float:
        """Exact P(A wins the series). 'both' averages the two starting-side assignments."""
        if start_blue == "both":
            return 0.5 * (self.series_win_prob(pol_A, pol_B, "A")
                          + self.series_win_prob(pol_A, pol_B, "B"))
        if not (pol_A.deterministic and pol_B.deterministic):
            raise ValueError("exact evaluation needs deterministic policies; use monte_carlo()")
        policies = {"A": pol_A, "B": pol_B}
        memo: Dict[SeriesState, float] = {}

        def rec(state: SeriesState) -> float:
            if self.is_over(state):
                return 1.0 if state.wins_A >= self.cfg.wins_needed else 0.0
            if state in memo:
                return memo[state]
            d = self.play_draft(state, policies, start_blue)
            v = (d.p_A * rec(self.next_state(state, d, True))
                 + (1 - d.p_A) * rec(self.next_state(state, d, False)))
            memo[state] = v
            return v

        return rec(self.initial_state())


# --------------------------------------------------------------------------- #
# 6. Demo
# --------------------------------------------------------------------------- #
def _demo(csv_path: str) -> None:
    eng = FearlessEngine.from_csv(csv_path)
    t = eng.table.sort_values("strength", ascending=False)
    print("Generic strength (log-odds, centred), top and bottom 5:")
    cols = ["Champion_Name", "Win_Rate", "Pick_Rate", "Ban_Rate", "strength"]
    print(t[cols].head(5).to_string(index=False))
    print("...")
    print(t[cols].tail(5).to_string(index=False))
    print(f"\nPool size N = {eng.n}")
    for bo in (3, 5):
        e = FearlessEngine(eng.strength, eng.names, GameConfig(best_of=bo))
        exact = e.series_win_prob(GENERIC_POLICY, WASTED_BAN_POLICY, "both")
        mc, se = e.monte_carlo(GENERIC_POLICY, RANDOM_BAN_POLICY, n=4000, seed=1)
        print(f"Bo{bo}: generic-ban vs wasted-ban  exact P(A wins) = {exact:.4f}"
              f"   |  generic-ban vs random-ban  MC P(A wins) = {mc:.4f} +/- {se:.4f}")
    rng = np.random.default_rng(7)
    res = eng.simulate_series(GENERIC_POLICY, GENERIC_POLICY, rng)
    print("\nOne sample Bo3 (generic vs generic):")
    for g in res.games:
        print(f"  Game {g.game_no} (blue={g.blue}): bans "
              f"A={eng.names[g.bans['A']]}, B={eng.names[g.bans['B']]}; picks "
              f"A={eng.names[g.picks['A']]}, B={eng.names[g.picks['B']]}; P(A)={g.p_A:.3f}")
    print(f"  Winner: {res.winner} ({res.wins_A}-{res.wins_B})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="All_Top_Lane_Champions_Worlds_2026.csv")
    _demo(ap.parse_args().csv)
