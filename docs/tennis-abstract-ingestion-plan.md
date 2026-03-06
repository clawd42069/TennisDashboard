# Tennis Abstract ingestion (v1)

Goal: import enough historical data to power Tier-1 features:
- Style matchup proxy features
- Opponent-quality progression
- Surface splits + recent form

## Data source
Jeff Sackmann (Tennis Abstract):
- ATP: https://github.com/JeffSackmann/tennis_atp
- WTA: https://github.com/JeffSackmann/tennis_wta

## What we import first (MVP)
1) **Matches (tour-level main draw)** for last N years (start with 2015+)
   - winner/loser ids
   - surface
   - score
   - tourney_date
   - round
   - basic serve/return stats if present

2) **Players** (id, name, hand, height, dob, country)

3) **Rankings** (weekly rankings) for opponent-quality classification

## Tables (SQLite: dashboard_v2/data/tennis.db)
- ta_players(player_id, first_name, last_name, hand, dob, country, height)
- ta_matches(match_id, tourney_date, tourney_name, surface, round,
  winner_id, loser_id, winner_rank, loser_rank, score,
  w_ace,w_df,w_svpt,w_1stIn,w_1stWon,w_2ndWon,w_SvGms,w_bpSaved,w_bpFaced,
  l_ace,l_df,l_svpt,l_1stIn,l_1stWon,l_2ndWon,l_SvGms,l_bpSaved,l_bpFaced)
- ta_rankings(ranking_date, rank, player_id, points)

## Feature views (built after import)
- surface_splits(player_id, surface, matches, win_pct, hold_pct_proxy, break_pct_proxy)
- recent_opponent_quality(player_id, as_of_date, last_n, avg_opp_rank, trend)
- style_proxy(player_id, as_of_date, serve_strength, return_strength, aggression_proxy)

## Next
After import, wire dashboard Matchup Report to compute:
- Style score (serve/return proxies + surface)
- Progression score (last N opponents quality ladder)
- Output: bias + confidence + reasons
