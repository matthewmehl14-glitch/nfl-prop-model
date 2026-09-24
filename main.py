import os
import json
import requests
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
FLAT_BET_SIZE = 25
MARKETS = "player_pass_yds,player_pass_completions,player_rush_yds,player_rush_attempts,player_rush_tds,player_reception_yds,player_receptions,player_reception_tds"

# 1. Load the Backtest Log
with open('backtest_log.json', 'r') as f:
    backtest_log = json.load(f)

# 2. Grade Pending Bets (Historical check)
def grade_pending_bets():
    # Pull current season weekly data to check actual results against logged bets
    # For a live environment, you would merge nfl.import_weekly_data([2026]) here
    for bet in backtest_log:
        if bet['status'] == 'pending':
            # Simulated Grading Logic Placeholder:
            # If actual > line for an Over:
            # bet['result'] = 'Win'
            # bet['pnl'] = FLAT_BET_SIZE * (decimal_odds - 1)
            # else: bet['pnl'] = -FLAT_BET_SIZE
            # bet['status'] = 'graded'
            pass

grade_pending_bets()

# 3. Fetch Data & Run 10k Simulations
def run_simulations():
    ui_cards = []
    
    # Placeholder for The Odds API request
    # odds_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/.../odds?markets={MARKETS}&apiKey={ODDS_API_KEY}"
    
    # Example simulated loop through pulled match-ups
    players = [
        {"name": "Patrick Mahomes", "team": "KC", "opp": "LV", "prop": "player_pass_yds", "line": 265.5, "odds": 1.909, "mean": 272.5, "std": 45.0},
        {"name": "Isiah Pacheco", "team": "KC", "opp": "LV", "prop": "player_rush_tds", "line": 0.5, "odds": 2.10, "mean": 0.6, "std": 0.0} # TDs use Poisson
    ]
    
    for p in players:
        # Monte Carlo 10,000 sims
        if "tds" in p['prop']:
            sims = np.random.poisson(lam=p['mean'], size=10000)
        else:
            sims = np.random.normal(loc=p['mean'], scale=p['std'], size=10000)
            
        hit_rate = np.sum(sims > p['line']) / 10000
        ev_pct = (hit_rate * p['odds']) - 1
        
        # 4. Log > 3% EV Plays for the $25 Flat Bet Backtest
        if ev_pct > 0.03:
            if ev_pct < 0.05: bucket = "3-5%"
            elif ev_pct < 0.08: bucket = "5-8%"
            else: bucket = "8%+"
            
            new_bet = {
                "date": str(datetime.now().date()),
                "game": f"{p['team']} vs {p['opp']}",
                "player": p['name'],
                "prop": p['prop'],
                "line": p['line'],
                "ev": round(ev_pct, 4),
                "bucket": bucket,
                "status": "pending",
                "pnl": 0
            }
            # Prevent duplicate logging on the same day
            if not any(b['player'] == p['name'] and b['prop'] == p['prop'] and b['date'] == new_bet['date'] for b in backtest_log):
                backtest_log.append(new_bet)
                
        # Build UI Data
        ui_cards.append({
            "game": f"{p['team']} vs {p['opp']}",
            "name": p['name'],
            "prop": p['prop'],
            "line": p['line'],
            "proj": round(np.median(sims), 1),
            "ev": round(ev_pct, 4)
        })
        
    return ui_cards

ui_data = run_simulations()

# 5. Write Updates to Files
with open('backtest_log.json', 'w') as f:
    json.dump(backtest_log, f, indent=4)

with open('data.json', 'w') as f:
    json.dump(ui_data, f, indent=4)
