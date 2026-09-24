import os
import json
import requests
import numpy as np
import pandas as pd
import re
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

def clean_name(name):
    """Normalizes player names for reliable dataset merging."""
    if not isinstance(name, str): return ""
    name = re.sub(r'(?i)\b(jr\.?|sr\.?|iii|ii|iv|v)\b', '', name)
    name = re.sub(r'[^\w\s]', '', name)
    return ' '.join(name.strip().lower().split())

def get_defensive_multipliers(weekly_df):
    """Calculates opponent defensive multipliers relative to league averages."""
    if weekly_df.empty: return {}
    league_pass_avg = weekly_df['passing_yards'].mean()
    league_rush_avg = weekly_df['rushing_yards'].mean()
    league_rec_avg = weekly_df['receiving_yards'].mean()
    
    def_stats = weekly_df.groupby('opponent_team').agg({
        'passing_yards': 'mean', 'rushing_yards': 'mean', 'receiving_yards': 'mean'
    }).reset_index()
    
    multipliers = {}
    for _, row in def_stats.iterrows():
        team = row['opponent_team']
        multipliers[team] = {
            'player_pass_yds': float(np.clip(row['passing_yards'] / max(league_pass_avg, 1), 0.85, 1.15)),
            'player_rush_yds': float(np.clip(row['rushing_yards'] / max(league_rush_avg, 1), 0.85, 1.15)),
            'player_reception_yds': float(np.clip(row['receiving_yards'] / max(league_rec_avg, 1), 0.85, 1.15))
        }
    return multipliers

def get_live_data():
    if not ODDS_API_KEY: return []
    ui_cards = []
    
    # Fetch schedule
    events_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events?apiKey={ODDS_API_KEY}"
    try:
        events_response = requests.get(events_url).json()
    except:
        return []
        
    # Fetch NFL Stats (Fallback to 2025 if 2026 data isn't fully published yet)
    current_year = 2026
    try:
        weekly = nfl.import_weekly_data([current_year])
        if weekly.empty: weekly = nfl.import_weekly_data([current_year - 1])
    except:
        try:
            weekly = nfl.import_weekly_data([current_year - 1])
        except:
            weekly = pd.DataFrame()

    def_multipliers = get_defensive_multipliers(weekly) if not weekly.empty else {}

    if not weekly.empty:
        stats = weekly.groupby('player_display_name').agg({
            'passing_yards': ['mean', 'std'], 'rushing_yards': ['mean', 'std'],
            'receiving_yards': ['mean', 'std'], 'passing_tds': ['mean'],
            'rushing_tds': ['mean'], 'receiving_tds': ['mean']
        }).reset_index()
        stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
        stats['clean_name'] = stats['player_display_name'].apply(clean_name)
    else:
        stats = pd.DataFrame()

    markets = "player_pass_yds,player_rush_yds,player_reception_yds,player_pass_tds,player_rush_tds,player_reception_tds"

    for event in events_response:
        game_title = f"{event['away_team']} @ {event['home_team']}"
        event_id = event['id']
        
        # Prioritize FanDuel, fallback to DraftKings
        odds_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds?apiKey={ODDS_API_KEY}&regions=us&markets={markets}&bookmakers=fanduel,draftkings"
        odds_response = requests.get(odds_url).json()
        
        if 'bookmakers' not in odds_response or not odds_response['bookmakers']:
            continue
            
        bookmaker = next((b for b in odds_response['bookmakers'] if b['key'] == 'fanduel'), odds_response['bookmakers'][0])
        
        for market in bookmaker.get('markets', []):
            prop_type = market['key']
            
            for outcome in market.get('outcomes', []):
                # We only need the Over/Yes side to run the raw EV calculation
                if outcome.get('name') not in ['Over', 'Yes']: 
                    continue
                
                player_name = outcome.get('description')
                line = outcome.get('point', 0.5)
                price = outcome.get('price', -110)
                
                # Convert American to Decimal
                dec_over = (price / 100) + 1 if price > 0 else (100 / abs(price)) + 1
                
                cleaned_api_name = clean_name(player_name)
                player_stats = stats[stats['clean_name'] == cleaned_api_name] if not stats.empty else pd.DataFrame()
                is_td = "tds" in prop_type
                
                # SAFE FALLBACK: If player data is missing/rookies, default to the sportsbook line 
                # so they STILL appear on the dashboard rather than being deleted.
                mean_val = line
                std_val = max(line * 0.25, 1.0)
                found_real_stats = False

                if not player_stats.empty:
                    try:
                        if prop_type == 'player_pass_yds':
                            mean_val = player_stats['passing_yards_mean'].values[0]
                            std_val = player_stats['passing_yards_std'].values[0]
                        elif prop_type == 'player_rush_yds':
                            mean_val = player_stats['rushing_yards_mean'].values[0]
                            std_val = player_stats['rushing_yards_std'].values[0]
                        elif prop_type == 'player_reception_yds':
                            mean_val = player_stats['receiving_yards_mean'].values[0]
                            std_val = player_stats['receiving_yards_std'].values[0]
                        elif prop_type == 'player_pass_tds':
                            mean_val = player_stats['passing_tds_mean'].values[0]
                        elif prop_type == 'player_rush_tds':
                            mean_val = player_stats['rushing_tds_mean'].values[0]
                        elif prop_type == 'player_reception_tds':
                            mean_val = player_stats['receiving_tds_mean'].values[0]

                        if not pd.isna(mean_val):
                            found_real_stats = True
                            if not is_td and (pd.isna(std_val) or std_val <= 0):
                                std_val = max(mean_val * 0.25, 1.0) # Handle 1-game sample sizes
                        else:
                            mean_val = line
                    except:
                        pass
                
                # Apply opponent adjustment only if we have real data
                def_mult = 1.0
                if found_real_stats:
                    opp_team_code = event['home_team'] if player_name in event['away_team'] else event['away_team']
                    if opp_team_code in def_multipliers and prop_type in def_multipliers[opp_team_code]:
                        def_mult = def_multipliers[opp_team_code][prop_type]
                        
                adjusted_mean = mean_val * def_mult

                # Run Distributions
                simulations = 10000
                if is_td:
                    sims = np.random.poisson(lam=max(adjusted_mean, 0.01), size=simulations)
                else:
                    if adjusted_mean <= 0 or std_val <= 0: continue
                    shape = (adjusted_mean / std_val) ** 2
                    scale = (std_val ** 2) / adjusted_mean
                    sims = np.random.gamma(shape=shape, scale=scale, size=simulations)
                    
                sim_hit_rate = np.sum(sims > line) / simulations
                projection = np.median(sims)
                
                # Straight EV Calculation (No Devig)
                ev_pct = (sim_hit_rate * dec_over) - 1.0
                
                if ev_pct > 0.03:
                    bucket = "3-5%" if ev_pct < 0.05 else ("5-8%" if ev_pct < 0.08 else "8%+")
                    new_bet = {
                        "date": str(datetime.now().date()),
                        "game": game_title,
                        "player": player_name,
                        "prop": prop_type,
                        "line": line,
                        "odds": price,
                        "ev": round(ev_pct, 4),
                        "bucket": bucket,
                        "stake": FLAT_BET_SIZE,
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
