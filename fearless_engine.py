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
    m      matchup (counter) term, antisymmetric: m[a,b] = -m[b,a] -> from the u.gg workbook
    pi_T   team T's top laner's proficiency on a champion         -> from the gol.gg workbook
    delta  team A's edge from the other four roles               -> 0 by default

Policies are objects with a ban function and a pick function, so strategies can be swapped
in and compared.  Deterministic policies can be evaluated EXACTLY (no sampling) with
FearlessEngine.series_win_prob(); any policy can be simulated with simulate_series().
"""
from __future__ import annotations

import argparse
import os
import re
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
    """m[i, j] = log-odds edge for the team on champion i vs champion j (antisymmetric).
    Zeros by default; MatchupData builds it from the u.gg workbook."""

    def __init__(self, n: int, matrix: Optional[np.ndarray] = None):
        m = np.zeros((n, n)) if matrix is None else np.asarray(matrix, dtype=float)
        if m.shape != (n, n):
            raise ValueError(f"matchup matrix must be {n}x{n}, got {m.shape}")
        if not np.allclose(m, -m.T):
            raise ValueError("matchup matrix must be antisymmetric: m[i,j] = -m[j,i]")
        self.m = m


class MatchupData:
    """Champion-vs-champion win rates (u.gg) turned into the antisymmetric matchup term m.

    Why not use logit(W[a,b]) directly?  A raw matchup win rate already contains each
    champion's general strength, and the game model adds s[a] - s[b] separately, so using it
    directly would count strength twice.  We split the table instead:

        logit W[a,b] = u[a] - u[b] + c[a,b]

    u[a] is champion a's mean log-odds against the whole pool (its strength in this table; the
    least-squares fit of strength differences) and c[a,b] is the COUNTER effect that is left
    over.  c is antisymmetric and every row sums to zero.  Only c enters the game model:

        m[a,b] = weight * clip(c[a,b], -cap, +cap)

    weight (default 0.5) and cap (default 0.30 log-odds, about 7 points) are assumptions: the
    workbook has no sample sizes, rarely played champions have extreme cells (some c exceed 1),
    and ladder (Master+) matchups may only partly transfer to pro play.  Sensitivity-test both.
    """

    def __init__(self, source, champion_names, weight: float = 0.5, cap: float = 0.30):
        W = pd.read_excel(source, sheet_name=0, index_col=0) if isinstance(source, str) else source.copy()
        rows = {_norm(c): c for c in W.index}
        cols = {_norm(c): c for c in W.columns}
        missing = [n for n in champion_names if _norm(n) not in rows or _norm(n) not in cols]
        if missing:
            raise KeyError(f"champions missing from matchup table: {missing}")
        W = W.loc[[rows[_norm(n)] for n in champion_names], [cols[_norm(n)] for n in champion_names]]
        n = len(champion_names)
        v = W.to_numpy(dtype=float)
        off = ~np.eye(n, dtype=bool)
        if np.isnan(v[off]).any():
            raise ValueError("matchup table has missing off-diagonal cells")
        self.max_asymmetry = float(np.abs(v + v.T - 1.0)[off].max())      # 0 if W[a,b] + W[b,a] = 1
        v = np.where(off, 0.5 * (v + 1.0 - v.T), 0.5)                      # enforce W[a,b] + W[b,a] = 1
        v = np.clip(v, 0.01, 0.99)
        L = np.log(v / (1.0 - v))
        np.fill_diagonal(L, 0.0)
        self.names = list(champion_names)
        self.params = dict(weight=weight, cap=cap)
        self.win_rate = pd.DataFrame(np.where(off, v, np.nan), index=self.names, columns=self.names)
        self.u = L.mean(axis=1)                                            # strength implied by the table
        self.counter = L - (self.u[:, None] - self.u[None, :])             # c[a,b]
        self.m = weight * np.clip(self.counter, -cap, cap)

    def top_counters(self, k: int = 5):
        """The k largest matchup edges as (champion, victim, m)."""
        iu = np.argwhere(self.m > 0)
        order = np.argsort(-self.m[iu[:, 0], iu[:, 1]])[:k]
        return [(self.names[iu[i, 0]], self.names[iu[i, 1]], float(self.m[iu[i, 0], iu[i, 1]])) for i in order]


class ProficiencyModel:
    """pi[team][c] = log-odds bonus for that team's top laner on champion c.
    Zeros by default; FearlessEngine.set_players() fills it from the gol.gg workbook."""

    def __init__(self, n: int):
        self.pi: Dict[str, np.ndarray] = {t: np.zeros(n) for t in TEAMS}

    def set(self, team: str, vector) -> None:
        v = np.asarray(vector, dtype=float)
        if v.shape != self.pi[team].shape:
            raise ValueError("proficiency vector has the wrong length")
        self.pi[team] = v



# --------------------------------------------------------------------------- #
# 2b. Player proficiency from the gol.gg workbook
# --------------------------------------------------------------------------- #
def _norm(name) -> str:
    """Match champion names across files ("K'Sante" == "Ksante")."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


@dataclass(frozen=True)
class ProficiencyParams:
    """Proficiency formula parameters (all log-odds units unless noted).

    pi = gamma_wr * logit(w_hat) + gamma_exp * ln(1 + g) / ln(1 + g_max),  clipped to +/- cap
    w_hat = (wins + k_prior * 0.5) / (g + k_prior)       win rate shrunk toward 50%
    wins  = win_rate * g.   A champion the player has never played has pi = 0 (neutral).
    """
    k_prior: float = 5.0      # pseudo-games at 50% that anchor a small sample
    gamma_wr: float = 0.40    # weight on the shrunk win rate
    gamma_exp: float = 0.30   # weight on experience (games played)
    g_max: float = 36.0       # games that map to experience = 1 (the most in the workbook)
    cap: float = 1.0          # |pi| never exceeds this


def proficiency_value(games, win_rate, p: ProficiencyParams = ProficiencyParams()):
    """Vectorised proficiency. games: NaN or 0 = never played. win_rate: fraction, NaN = unknown."""
    g = np.nan_to_num(np.asarray(games, dtype=float), nan=0.0)
    wr = np.asarray(win_rate, dtype=float)
    wr = np.where(np.isnan(wr), 0.5, wr)                 # unknown outcome -> no win-rate signal
    w_hat = (wr * g + 0.5 * p.k_prior) / (g + p.k_prior)
    pi = (p.gamma_wr * np.log(w_hat / (1.0 - w_hat))
          + p.gamma_exp * np.log1p(g) / np.log1p(p.g_max))
    return np.where(g > 0, np.clip(pi, -p.cap, p.cap), 0.0)


class PlayerProficiency:
    """Loads the workbook ('Win Rates' and 'Games Played' sheets) and computes pi[player, champion].

    The 'Games Played' sheet decides whether a champion was played (blank = never = neutral 0).
    A win rate with no games listed is ignored.  Data oddities are collected in self.issues.
    """

    def __init__(self, path: str, champion_names, params: Optional[ProficiencyParams] = None):
        self.params = params or ProficiencyParams()
        wr = pd.read_excel(path, "Win Rates", index_col=0)
        gp = pd.read_excel(path, "Games Played", index_col=0)
        lookup = {_norm(c): c for c in wr.columns}
        missing = [n for n in champion_names if _norm(n) not in lookup]
        if missing:
            raise KeyError(f"champions missing from workbook: {missing}")
        cols = [lookup[_norm(n)] for n in champion_names]
        wr = wr[cols].astype(float); gp = gp.loc[wr.index, cols].astype(float)
        wr.columns = gp.columns = list(champion_names)
        self.win_rate, self.games = wr, gp
        k = (wr * gp).round()
        off = ((k / gp).round(3) - wr).abs() > 0.0011
        self.issues = {
            "win_rate_but_no_games": [(p, c) for p in wr.index for c in wr.columns
                                      if pd.notna(wr.loc[p, c]) and pd.isna(gp.loc[p, c])],
            "games_but_no_win_rate": [(p, c) for p in wr.index for c in wr.columns
                                      if pd.isna(wr.loc[p, c]) and pd.notna(gp.loc[p, c])],
            "win_rate_not_whole_wins": [(p, c) for p in wr.index for c in wr.columns
                                        if pd.notna(wr.loc[p, c]) and pd.notna(gp.loc[p, c]) and off.loc[p, c]],
        }
        self.table = pd.DataFrame(proficiency_value(gp.to_numpy(), wr.to_numpy(), self.params),
                                  index=wr.index, columns=wr.columns)

    @property
    def players(self) -> List[str]:
        return list(self.table.index)

    def vector(self, player: str) -> np.ndarray:
        if player not in self.table.index:
            raise KeyError(f"unknown player '{player}'; choose from {self.players}")
        return self.table.loc[player].to_numpy()

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


def make_targeted_ban(own_weight: float = 0.0) -> PolicyFn:
    """Opponent-targeted ban (a one-ply denial heuristic).

    threat(c) = the opponent's guaranteed game win probability if they pick champion c and we
    reply with our best response from what is left (the minimum over our replies of P_opp[c, r]).
    It uses the opponent's proficiency and the matchup term, so it is specific to who they are.
    The ban is the legal champion, available to the opponent, with the highest threat.

    own_weight > 0 subtracts own_weight * (our own guaranteed value on c) from the score, so a
    champion that is just as good for us is less attractive to ban.  0 is pure denial.
    It looks one pick ahead only: no opponent-ban forecasting and no look ahead to later games
    (that is what the equilibrium benchmark will add).
    """
    def worst_case(P: np.ndarray, replier_available: np.ndarray, n: int) -> np.ndarray:
        reply = replier_available[None, :] & ~np.eye(n, dtype=bool)        # r available and r != c
        low = np.where(reply, P, np.inf).min(axis=1)
        return np.where(np.isinf(low), 1.0, low)

    def targeted_ban(engine, view):
        me, opp, n = view.team, other(view.team), engine.n
        opp_av = np.zeros(n, dtype=bool)
        opp_av[engine.legal_picks(view.state, opp, view.banned, ())] = True
        me_av = np.zeros(n, dtype=bool)
        me_av[engine.legal_picks(view.state, me, view.banned, ())] = True
        cands = np.array([c for c in view.legal if opp_av[c]], dtype=int)
        if len(cands) == 0:                                    # nothing the opponent could pick
            return generic_ban(engine, view)
        score = worst_case(engine.prob_matrix(opp), me_av, n)  # opponent picks c, we reply
        if own_weight:
            score = score - own_weight * worst_case(engine.prob_matrix(me), opp_av, n)
        return int(cands[np.argmax(score[cands])])

    return targeted_ban


targeted_ban = make_targeted_ban()


def targeted_policy(own_weight: float = 0.0) -> Policy:
    tag = "" if own_weight == 0 else f" (own_weight={own_weight:g})"
    return Policy(f"targeted-ban{tag} + best-response pick", make_targeted_ban(own_weight), best_response_pick)


GENERIC_POLICY = Policy("generic-ban + best-response pick", generic_ban, best_response_pick)
TARGETED_POLICY = targeted_policy(0.0)
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
        self.players: Optional[PlayerProficiency] = None
        self.matchup_data: Optional[MatchupData] = None
        self.refresh()

    @classmethod
    def from_csv(cls, path: str, config: Optional[GameConfig] = None,
                 shrink_k: float = 2.0, pop_weight: float = 0.015,
                 player_xlsx: Optional[str] = None,
                 proficiency_params: Optional[ProficiencyParams] = None,
                 matchup_xlsx: Optional[str] = None,
                 matchup_weight: float = 0.5, matchup_cap: float = 0.30, **kw):
        df = load_champions(path, shrink_k, pop_weight)
        names = df["Champion_Name"].tolist()
        md = MatchupData(matchup_xlsx, names, matchup_weight, matchup_cap) if matchup_xlsx else None
        if md is not None and "matchups" not in kw:
            kw["matchups"] = MatchupModel(len(names), md.m)
        eng = cls(df["strength"].to_numpy(), names, config, **kw)
        eng.matchup_data = md
        eng.table = df
        eng.players = (PlayerProficiency(player_xlsx, eng.names, proficiency_params)
                       if player_xlsx else None)
        return eng

    def set_players(self, player_A: str, player_B: str) -> None:
        """Use these two players' proficiencies for team A's and team B's top laners."""
        if getattr(self, "players", None) is None:
            raise RuntimeError("no player workbook loaded; pass player_xlsx to from_csv()")
        self.proficiency.set("A", self.players.vector(player_A))
        self.proficiency.set("B", self.players.vector(player_B))
        self.refresh()

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


def _preview(eng) -> None:
    """Preliminary look: team A switches from generic to targeted bans, team B stays generic.
    Averaged over every ordered pair of players in the workbook (exact series probabilities,
    both starting sides).  Individuals stand in for teams, so this is a smoke test, not a result."""
    pp = eng.players
    variants = {"targeted, pure denial": targeted_policy(0.0), "targeted, own_weight=1": targeted_policy(1.0)}
    print("\nPreview: A switches generic -> targeted ban, B stays generic (all 240 ordered player pairs)")
    for bo in (3, 5):
        e = FearlessEngine(eng.strength, eng.names, GameConfig(best_of=bo), matchups=eng.matchups)
        gains = {k: [] for k in variants}
        for a in pp.players:
            for b in pp.players:
                if a == b:
                    continue
                e.proficiency.set("A", pp.vector(a)); e.proficiency.set("B", pp.vector(b)); e.refresh()
                base = e.series_win_prob(GENERIC_POLICY, GENERIC_POLICY, "both")
                for k, pol in variants.items():
                    gains[k].append(e.series_win_prob(pol, GENERIC_POLICY, "both") - base)
        for k, g in gains.items():
            g = np.array(g) * 100
            print(f"  Bo{bo} {k:<24} mean gain {g.mean():+.2f} pts | better in {np.mean(g > 1e-9):.0%} of pairs,"
                  f" worse in {np.mean(g < -1e-9):.0%} (max {g.max():+.2f}, min {g.min():+.2f})")


# --------------------------------------------------------------------------- #
# 6. Demo
# --------------------------------------------------------------------------- #
def _demo(csv_path: str, xlsx_path: Optional[str] = None, matchup_path: Optional[str] = None) -> None:
    eng = FearlessEngine.from_csv(csv_path, player_xlsx=xlsx_path, matchup_xlsx=matchup_path)
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
    if eng.matchup_data is not None:
        md = eng.matchup_data
        print(f"\nMatchups: weight={md.params['weight']}, cap={md.params['cap']}, "
              f"table asymmetry before enforcing W[a,b]+W[b,a]=1: {md.max_asymmetry:.4f}")
        print("  Largest edges: " + "; ".join(f"{a} over {b} ({m:+.3f})" for a, b, m in md.top_counters(4)))
        plain = FearlessEngine(eng.strength, eng.names, eng.cfg)

        def counter_value(e):                    # mean P(second picker wins) when it best-responds
            return float(np.mean([max(1 - e.P_A[a, b] for b in range(e.n) if b != a) for a in range(e.n)]))
        print(f"  Mean win prob of the best counterpick: {counter_value(plain):.4f} without matchups, "
              f"{counter_value(eng):.4f} with them")
    if eng.players is not None:
        pp = eng.players
        print(f"\nPlayer proficiency: {len(pp.players)} players; data notes:")
        for k, v in pp.issues.items():
            print(f"  {k}: {len(v)} cells")
        a, b = pp.players[1], pp.players[6]
        eng.set_players(a, b)
        for who in (a, b):
            top = pp.table.loc[who].sort_values(ascending=False).head(3)
            print(f"  {who} best: " + ", ".join(f"{c} ({v:+.2f})" for c, v in top.items()))
        for bo in (3, 5):
            e2 = FearlessEngine(eng.strength, eng.names, GameConfig(best_of=bo),
                                matchups=eng.matchups, proficiency=eng.proficiency)
            ex = e2.series_win_prob(GENERIC_POLICY, WASTED_BAN_POLICY, "both")
            print(f"  Bo{bo}, {a} (generic ban) vs {b} (wasted ban): exact P({a} wins) = {ex:.4f}")
    if eng.players is not None and eng.matchup_data is not None:
        _preview(eng)
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
    ap.add_argument("--xlsx", default=None, help="gol.gg player workbook for proficiency")
    ap.add_argument("--matchups", default=None, help="u.gg champion matchup workbook")
    args = ap.parse_args()
    _demo(args.csv, args.xlsx, args.matchups)
