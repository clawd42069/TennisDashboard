# Tennis Analyst — Build Checklist (2026-03-06)

Goal: ensure every spec decision from today is implemented in **dashboard_v2** once Ryan is back at his local machine.

---

## 0) Scope lock
- [ ] **ATP only** (ignore WTA unless explicitly requested)
- [ ] Markets supported: **Moneyline + Spreads + Totals**
- [ ] Output: **ranked shortlist** (not a single pick)
- [ ] Paper sizing: default **1u**; tiered **2–5u** based on EV + confidence

---

## 1) Data ingestion & refresh loop
- [ ] Default refresh interval set to **120s**
  - [ ] Configurable (easy switch to 60s later)
- [ ] On each refresh:
  - [ ] Pull current slate of upcoming/active ATP matches
  - [ ] Pull odds for ML/spreads/totals (store book + price + line)
  - [ ] Generate **Top N = 10** ranked bet candidates
  - [ ] Persist snapshot + candidates to DB

---

## 2) Database schema (SQLite)
### 2.1 Snapshots
- [ ] `snapshots` table created
  - fields (suggested): `id`, `created_at`, `tour` (ATP), `tournament`, `surface`, `source`, `model_version`, `refresh_interval_sec`
  - [ ] Store raw odds payload once per snapshot (e.g., `raw_odds_json`)
  - [ ] Optional: compress raw odds JSON (if size grows)

### 2.2 Ranked candidates (Top 10 per snapshot)
- [ ] `ranked_candidates` table created (1 row per candidate per snapshot)
  - identifiers: `snapshot_id`, `match_id`, `player_a`, `player_b`
  - market: `market_type` (ML|SPREAD|TOTAL), `side`, `line`, `price`, `book`
  - model outputs:
    - [ ] `p0` (baseline win prob from surface Match Elo)
    - [ ] `p_final` (after overlays + cap)
    - [ ] `q_implied` (market implied probability)
    - [ ] `ev` (expected value)
    - [ ] `confidence` (0–1)
    - [ ] `ev_adj` (= ev * confidence)
  - diagnostics / explainability:
    - [ ] `delta_elo_surface`
    - [ ] `delta_sr`
    - [ ] `delta_recency`
    - [ ] `delta_z_raw`, `delta_z_capped`
    - [ ] `matchup_flags_json` (list)
    - [ ] `units_suggested` (1/2/3/5) + `tier`

---

## 3) Probability engine (Elo-first)
### 3.1 Baseline probability from surface Match Elo
- [ ] Compute `ΔE = Elo_match_surface(A) - Elo_match_surface(B)`
- [ ] Convert to win prob:
  - `p0 = 1 / (1 + 10^(-ΔE/400))`
- [ ] Allow tuning of the 400 scale later (config constant)

### 3.2 Overlays (moderate)
- [ ] Convert p0 to log-odds: `z0 = log(p0/(1-p0))`
- [ ] Compute overlay delta (SR + Recency + MatchupFlags + Mods): `Δz`
- [ ] **Moderate cap**: `Δz_capped = clip(Δz, -0.6, +0.6)`
- [ ] Final: `p = sigmoid(z0 + Δz_capped)`

### 3.3 Bo3 vs Bo5 conversion (separate conversion)
- [ ] Default compute probability on Bo3 scale
- [ ] Back out per-set win probability `s` by solving:
  - `p_bo3 = s^2 * (3 - 2s)`
- [ ] Convert to Bo5:
  - `p_bo5 = s^3 * (10 - 15s + 6s^2)`
- [ ] Apply this conversion when match format is Bo5 (slams)

---

## 4) Serve/Return ratings (by surface) + shrinkage
### 4.1 Ratings to maintain (per player)
For each surface in **{hard, clay, grass}** (no indoor split for now):
- [ ] `E_match_surface`
- [ ] `E_serve_surface`
- [ ] `E_return_surface`
- [ ] Overall (all-surface) versions for shrinkage: `E_match_all`, `E_serve_all`, `E_return_all`

### 4.2 Shrinkage rule (surface → overall)
- [ ] For each component c ∈ {match, serve, return}:
  - `w = n_surface / (n_surface + k)`
  - `E_eff = w*E_surface_raw + (1-w)*E_all`
- [ ] Choose initial k constants (configurable):
  - Hard k=15, Clay k=12, Grass k=8 (tunable)
- [ ] Persist `n_surface` and `w` and show in UI for transparency

### 4.3 Hold/Break projections (foundation for spreads/totals)
- [ ] Implement functions:
  - `P_hold_A = f(E_serve_A_surface - E_return_B_surface)`
  - `P_hold_B = f(E_serve_B_surface - E_return_A_surface)`
- [ ] Start with a simple logistic mapping; tune later with backtests

---

## 5) EV + confidence + unit sizing
### 5.1 EV computation
- [ ] Convert odds to implied probability: `q = 1/decimal_odds` (store both)
- [ ] EV per 1 unit risked:
  - `EV = p*(d-1) - (1-p)`
- [ ] Rank by `EV_adj = EV * confidence`

### 5.2 Confidence score (0–1)
- [ ] Derive confidence from:
  - surface sample/shrinkage weight w
  - model agreement (Elo vs overlays direction)
  - market quality (book count/dispersion)
- [ ] Penalize when overlay repeatedly hits cap (signals overreach)

### 5.3 Units (1u / 2–5u)
- [ ] Output `units_suggested` based on **EV + confidence** (rule-based)
- [ ] Enforce: no **5u** unless confidence is high (threshold TBD)

---

## 6) UI requirements (dashboard)
- [ ] Matchup Report tab shows:
  - [ ] Top 10 ranked candidates
  - [ ] For each: market, line, price, book, p0, p_final, q, EV, confidence, EV_adj, units
  - [ ] Drilldown: raw odds JSON + contributor breakdown (ΔElo, ΔSR, ΔRecency, flags, Δz)
- [ ] Snapshots list/history view (so you can review earlier states)
- [ ] Paper tracker integrates units + market details and ties back to candidate/snapshot id

---

## 7) Logging / evaluation hooks (for later totals bias decision)
- [ ] Log every top-10 candidate each refresh (not every book line)
- [ ] For totals, store directional pick (Over/Under) and the modeled probability (when implemented)
- [ ] Record closing line/price later for CLV (optional phase 2)

---

## 8) Implementation order (recommended)
1) [ ] DB tables + migrations
2) [ ] Refresh loop (120s) + odds snapshot persistence
3) [ ] Elo-first p0 + Bo5 conversion + EV ranking (even if overlays = 0 initially)
4) [ ] UI: Top 10 + drilldown + snapshot history
5) [ ] Serve/Return ratings scaffolding + shrinkage + confidence
6) [ ] Spread/Total deeper modeling once hold/break is validated

---

## 9) Open questions (leave configurable)
- [ ] Elo scale constant (default 400)
- [ ] Overlay cap (default ±0.6)
- [ ] Shrinkage k per surface
- [ ] Top N (default 10)
- [ ] Refresh interval (default 120s)
