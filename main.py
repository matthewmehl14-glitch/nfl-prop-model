import os
import json
import requests
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
FLAT_BET_SIZE = 25

# 1. Load the Backtest Log
try:
    with open('backtest_log.json', 'r') as f:
        backtest_log = json.load(f)
except FileNotFoundError:
    backtest_log = []

def get_live_data():
    if not ODDS_API_KEY:
        print("No API Key found. Returning empty data.")
        return []

    ui_cards = []
    
    # Fetch current week's games
    events_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events?apiKey={ODDS_API_KEY}"
    events_response = requests.get(events_url).json()
    
    # Fetch 2026 player stats for baselines
    try:
        yearly_stats = nfl.import_seasonal_data([2026])
    except:
        yearly_stats = pd.DataFrame() # Fallback if nfl_data_py fails
        
    markets = "player_pass_yds,player_rush_yds,player_reception_yds"
    
    for event in events_response:
        game_title = f"{event['away_team']} @ {event['home_team']}"
        event_id = event['id']
        
        # Fetch live lines for this specific game
        odds_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds?apiKey={ODDS_API_KEY}&regions=us&markets={markets}&bookmakers=draftkings"
        odds_response = requests.get(odds_url).json()
        
        if 'bookmakers' not in odds_response or not odds_response['bookmakers']:
            continue
            
        bookmaker = odds_response['bookmakers'][0]
        for market in bookmaker['markets']:
            prop_type = market['key']
            
            for outcome in market['outcomes']:
                # The Odds API lists Over/Under. We only need one to get the line.
                if outcome['name'] == 'Over':
                    player_name = outcome['description']
                    line = outcome['point']
                    decimal_odds = 1.909 # Standard -110 fallback for calculation
                    if 'price' in outcome:
                        # Convert American to Decimal if needed, assuming API returns American or Decimal based on settings
                        decimal_odds = outcome['price'] if outcome['price'] < 100 else (outcome['price']/100)+1
                    
                    # Try to find player in real stats to get mean/std, otherwise fallback to rough estimates for the simulation to run
                    player_mean = line + 2.0 
                    player_std = line * 0.25 
                    
                    if not yearly_stats.empty and 'player_name' in yearly_stats.columns:
                        match = yearly_stats[yearly_stats['player_name'] == player_name]
                        if not match.empty:
                            if prop_type == 'player_pass_yds':
                                player_mean = match['passing_yards'].values[0] / match['games'].values[0]
                            elif prop_type == 'player_rush_yds':
                                player_mean = match['rushing_yards'].values[0] / match['games'].values[0]
                            elif prop_type == 'player_reception_yds':
                                player_mean = match['receiving_yards'].values[0] / match['games'].values[0]
                    
                    # Run 10k Monte Carlo Sims
                    sims = np.random.normal(loc=player_mean, scale=player_std, size=10000)
                    hit_rate = np.sum(sims > line) / 10000
                    ev_pct = (hit_rate * decimal_odds) - 1
                    projection = np.median(sims)
                    
                    # Log to Backtest if EV > 3%
                    if ev_pct > 0.03:
                        bucket = "3-5%" if ev_pct < 0.05 else ("5-8%" if ev_pct < 0.08 else "8%+")
                        new_bet = {
                            "date": str(datetime.now().date()),
                            "game": game_title,
                            "player": player_name,
                            "prop": prop_type,
                            "line": line,
                            "ev": round(ev_pct, 4),
                            "bucket": bucket,
                            "status": "pending",
                            "pnl": 0
                        }
                        if not any(b['player'] == player_name and b['prop'] == prop_type and b['date'] == new_bet['date'] for b in backtest_log):
                            backtest_log.append(new_bet)
                    
                    ui_cards.append({
                        "game": game_title,
                        "name": player_name,
                        "prop": prop_type,
                        "line": line,
                        "proj": round(projection, 1),
                        "ev": round(ev_pct, 4)
                    })
                    
    return ui_cards

ui_data = get_live_data()

with open('backtest_log.json', 'w') as f:
    json.dump(backtest_log, f, indent=4)

with open('data.json', 'w') as f:
    json.dump(ui_data, f, indent=4)
