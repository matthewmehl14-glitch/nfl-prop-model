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

def power_devig(p_over_implied, p_under_implied):
    """
    Solves p_over^k + p_under^k = 1 via bisection (Power Method).
    Removes vig while correcting for favorite-longshot bias in 2-way prop markets.
    """
    if p_over_implied <= 0 or p_under_implied <= 0:
        return p_over_implied / (p_over_implied + p_under_implied)
    
    low, high = 0.001, 10.0
    for _ in range(30):
        mid = (low + high) / 2.0
        val = (p_over_implied ** mid) + (p_under_implied ** mid)
        if val > 1.0:
            low = mid
        else:
            high = mid
            
    k = (low + high) / 2.0
    return p_over_implied ** k

def get_defensive_multipliers(weekly_df):
    """Calculates opponent defensive multipliers relative to league averages."""
    if weekly_df.empty:
        return {}
    
    # Calculate league averages per player appearance
    league_pass_avg = weekly_df['passing_yards'].mean()
    league_rush_avg = weekly_df['rushing_yards'].mean()
    league_rec_avg = weekly_df['receiving_yards'].mean()
    
    # Group by defending team (opponent_team)
    def_stats = weekly_df.groupby('opponent_team').agg({
        'passing_yards': 'mean',
        'rushing_yards': 'mean',
        'receiving_yards': 'mean'
    }).reset_index()
    
    multipliers = {}
    for _, row in def_stats.iterrows():
        team = row['opponent_team']
        multipliers[team] = {
            'player_pass_yds': float(np.clip(row['passing_yards'] / max(league_pass_avg, 1), 0.80, 1.25)),
            'player_rush_yds': float(np.clip(row['rushing_yards'] / max(league_rush_avg, 1), 0.80, 1.25)),
            'player_reception_yds': float(np.clip(row['receiving_yards'] / max(league_rec_avg, 1), 0.80, 1.25)),
            'player_rush_tds': 1.0,
            'player_reception_tds': 1.0,
            'player_pass_tds': 1.0
        }
    return multipliers

def get_live_data():
    if not ODDS_API_KEY:
        print("No API Key found. Exiting.")
        return []

    ui_cards = []
    
    # 2. Fetch current week's schedule
    events_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events?apiKey={ODDS_API_KEY}"
    try:
        events_response = requests.get(events_url).json()
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return []
        
    # 3. Pull seasonal and weekly stats
    current_year = 2026
    try:
        weekly = nfl.import_weekly_data([current_year])
    except Exception:
        try:
            weekly = nfl.import_weekly_data([current_year - 1])
        except Exception:
            weekly = pd.DataFrame()

    def_multipliers = get_defensive_multipliers(weekly) if not weekly.empty else {}

    # Player offensive baselines
    if not weekly.empty:
        stats = weekly.groupby('player_display_name').agg({
            'passing_yards': ['mean', 'std'],
            'rushing_yards': ['mean', 'std'],
            'receiving_yards': ['mean', 'std'],
            'passing_tds': ['mean'],
            'rushing_tds': ['mean'],
            'receiving_tds': ['mean']
        }).reset_index()
        stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
        stats['clean_name'] = stats['player_display_name'].apply(clean_name)
    else:
        stats = pd.DataFrame()

    markets = "player_pass_yds,player_rush_yds,player_reception_yds,player_pass_tds,player_rush_tds,player_reception_tds"

    for event in events_response:
        game_title = f"{event['away_team']} @ {event['home_team']}"
        event_id = event['id']
        
        # Pull FanDuel odds specifically
        odds_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds?apiKey={ODDS_API_KEY}&regions=us&markets={markets}&bookmakers=fanduel"
        odds_response = requests.get(odds_url).json()
        
        if 'bookmakers' not in odds_response or not odds_response['bookmakers']:
            continue
            
        bookmaker = odds_response['bookmakers'][0] # FanDuel
        
        for market in bookmaker.get('markets', []):
            prop_type = market['key']
            
            # Map outcomes by (player, line) to pair Over and Under together
            paired_outcomes = {}
            for outcome in market.get('outcomes', []):
                p_name = outcome.get('description')
                point = outcome.get('point', 0.5)
                side = outcome.get('name')
                price = outcome.get('price', -110)
                
                key = (p_name, point)
                if key not in paired_outcomes:
                    paired_outcomes[key] = {}
                paired_outcomes[key][side] = price
            
            # Evaluate paired markets
            for (player_name, line), sides in paired_outcomes.items():
                if 'Over' not in sides or 'Under' not in sides:
                    continue # Power de-vigging requires both sides of the 2-way market
                
                over_price = sides['Over']
                under_price = sides['Under']
                
                # Convert American to Decimal
                dec_over = (over_price / 100) + 1 if over_price > 0 else (100 / abs(over_price)) + 1
                dec_under = (under_price / 100) + 1 if under_price > 0 else (100 / abs(under_price)) + 1
                
                # Implied probabilities with vig
                imp_over = 1.0 / dec_over
                imp_under = 1.0 / dec_under
                
                # 4. Power De-vig FanDuel Line
                fd_fair_prob = power_devig(imp_over, imp_under)
                
                # Match player stats
                cleaned_api_name = clean_name(player_name)
                player_stats = stats[stats['clean_name'] == cleaned_api_name] if not stats.empty else pd.DataFrame()
                
                if player_stats.empty:
                    continue
                
                mean_val, std_val = 0, 0
                is_td = False
                
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
                        is_td = True
                    elif prop_type == 'player_rush_tds':
                        mean_val = player_stats['rushing_tds_mean'].values[0]
                        is_td = True
                    elif prop_type == 'player_reception_tds':
                        mean_val = player_stats['receiving_tds_mean'].values[0]
                        is_td = True
                except:
                    continue
                    
                if pd.isna(mean_val) or (not is_td and pd.isna(std_val)):
                    continue
                
                # 5. Apply Opponent Defensive Multiplier
                # Default factor = 1.0 if team name matching isn't direct
                def_mult = 1.0
                opp_team_code = event['home_team'] if player_name in event['away_team'] else event['away_team']
                if opp_team_code in def_multipliers and prop_type in def_multipliers[opp_team_code]:
                    def_mult = def_multipliers[opp_team_code][prop_type]
                    
                adjusted_mean = mean_val * def_mult

                # 6. Run 10,000 Monte Carlo Simulations
                simulations = 10000
                if is_td or "tds" in prop_type:
                    sims = np.random.poisson(lam=max(adjusted_mean, 0.01), size=simulations)
                else:
                    if adjusted_mean <= 0 or std_val <= 0:
                        continue
                    shape = (adjusted_mean / std_val) ** 2
                    scale = (std_val ** 2) / adjusted_mean
                    sims = np.random.gamma(shape=shape, scale=scale, size=simulations)
                    
                sim_hit_rate = np.sum(sims > line) / simulations
                projection = np.median(sims)
                
                # 7. Consensus Shrinkage Blend (60% Sim, 40% De-vigged FanDuel)
                blended_prob = (0.60 * sim_hit_rate) + (0.40 * fd_fair_prob)
                
                # EV calculation against the available retail price
                ev_pct = (blended_prob * dec_over) - 1.0
                
                # 8. Log Qualified Plays (>3% EV) for $25 Flat Bet Tracking
                if ev_pct > 0.03:
                    bucket = "3-5%" if ev_pct < 0.05 else ("5-8%" if ev_pct < 0.08 else "8%+")
                    new_bet = {
                        "date": str(datetime.now().date()),
                        "game": game_title,
                        "player": player_name,
                        "prop": prop_type,
                        "line": line,
                        "odds": over_price,
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
