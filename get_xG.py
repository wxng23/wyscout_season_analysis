import requests
import pandas as pd
import os
from datetime import datetime
import ast
from dotenv import load_dotenv

load_dotenv()

api_version = 3
WYSCOUT_API_USERNAME = os.getenv('WYSCOUT_API_USERNAME')
WYSCOUT_API_PASSWORD = os.getenv('WYSCOUT_API_PASSWORD')
BASE_PATH = os.getenv('BASE_PATH')
SEASON_DATA_PATH = os.getenv('SEASON_DATA_PATH')
verbose = True

def process_regular_season_xg(input_csv, output_filename="big10_xg.csv"):
    df = pd.read_csv(input_csv)

    # 1. Parse columns
    def safe_eval(val):
        if pd.isna(val) or val == "" or not isinstance(val, str): return {}
        try: return ast.literal_eval(val)
        except: return {}

    df['team_name'] = df['team'].apply(lambda x: safe_eval(x).get('name'))
    df['opp_name'] = df['opponentTeam'].apply(lambda x: safe_eval(x).get('name'))
    
    df['xg_val'] = df['shot'].apply(lambda x: safe_eval(x).get('xg', 0))
    df['is_goal'] = df['shot'].apply(lambda x: 1 if safe_eval(x).get('isGoal') is True else 0)
    
    df['event_primary'] = df['type'].apply(lambda x: safe_eval(x).get('primary'))
    shots_df = df[df['event_primary'] == 'shot'].copy()

    stats_for = shots_df.groupby('team_name').agg({
        'xg_val': 'sum',
        'is_goal': 'sum'
    }).reset_index().rename(columns={'team_name': 'Team', 'xg_val': 'Total_xG_For', 'is_goal': 'Total_G_For'})

    stats_against = shots_df.groupby('opp_name').agg({
        'xg_val': 'sum',
        'is_goal': 'sum'
    }).reset_index().rename(columns={'opp_name': 'Team', 'xg_val': 'Total_xG_Against', 'is_goal': 'Total_G_Against'})

    matches_played = df.groupby('team_name')['matchId'].nunique().reset_index()
    matches_played.columns = ['Team', 'Games_Played']

    summary = pd.merge(stats_for, stats_against, on='Team')
    summary = pd.merge(summary, matches_played, on='Team')

    summary['xG_For_Per_Game'] = summary['Total_xG_For'] / summary['Games_Played']
    summary['xG_Against_Per_Game'] = summary['Total_xG_Against'] / summary['Games_Played']

    final_cols = [
        'Team', 
        'xG_For_Per_Game', 
        'xG_Against_Per_Game', 
        'Total_xG_For', 
        'Total_G_For', 
        'Total_xG_Against', 
        'Total_G_Against', 
        'Games_Played'
    ]
    
    summary = summary[final_cols]
    
    summary.to_csv(output_filename, index=False)

def process_formation_stats(event_csv, league_csv, output_filename="michigan_formation_stats.csv"):
    df = pd.read_csv(event_csv)
    league_df = pd.read_csv(league_csv)
    
    mean_xg_for = league_df['xG_For_Per_Game'].mean()
    mean_xg_against = league_df['xG_Against_Per_Game'].mean()

    def safe_eval(val):
        if pd.isna(val) or val == "" or not isinstance(val, str): return {}
        try: return ast.literal_eval(val)
        except: return {}

    df['team_name'] = df['team'].apply(lambda x: safe_eval(x).get('name'))
    df['opp_name'] = df['opponentTeam'].apply(lambda x: safe_eval(x).get('name'))
    df['formation'] = df['team'].apply(lambda x: safe_eval(x).get('formation'))
    df['opp_formation'] = df['opponentTeam'].apply(lambda x: safe_eval(x).get('formation'))
    df['xg_val'] = df['shot'].apply(lambda x: safe_eval(x).get('xg', 0))
    df['is_goal'] = df['shot'].apply(lambda x: 1 if safe_eval(x).get('isGoal') is True else 0)
    
    df = df.merge(league_df[['Team', 'xG_Against_Per_Game']], left_on='opp_name', right_on='Team', how='left')
    df = df.rename(columns={'xG_Against_Per_Game': 'Opp_Def_Avg'})
    
    df = df.merge(league_df[['Team', 'xG_For_Per_Game']], left_on='team_name', right_on='Team', how='left', suffixes=('', '_opp'))
    df = df.rename(columns={'xG_For_Per_Game': 'Opp_Off_Avg'})

    # Adj xG For = xG / (Opponent's defensive quality relative to average)
    df['adj_xg_for'] = df['xg_val'] / (df['Opp_Def_Avg'] / mean_xg_against)
    # Adj xG Against = xG / (Opponent's offensive quality relative to average)
    df['adj_xg_against'] = df['xg_val'] / (df['Opp_Off_Avg'] / mean_xg_for)

    mich_for = df[df['team_name'] == 'Michigan Wolverines'].groupby('formation').agg({
        'adj_xg_for': 'sum',
        'xg_val': 'sum',
        'is_goal': 'sum'
    }).rename(columns={'xg_val': 'Total_xG_For', 'is_goal': 'Total_G_For'})

    mich_against = df[df['team_name'] != 'Michigan Wolverines'].groupby('opp_formation').agg({
        'adj_xg_against': 'sum',
        'xg_val': 'sum',
        'is_goal': 'sum'
    }).rename(columns={'xg_val': 'Total_xG_Against', 'is_goal': 'Total_G_Against'})

    games_per_form = df[df['team_name'] == 'Michigan Wolverines'].groupby('formation')['matchId'].nunique().to_frame('Games_Played')

    summary = games_per_form.join(mich_for).join(mich_against).fillna(0)
    
    summary['Adj_xG_For_Per_Game'] = summary['adj_xg_for'] / summary['Games_Played']
    summary['Adj_xG_Against_Per_Game'] = summary['adj_xg_against'] / summary['Games_Played']
    summary['xG_For_Per_Game'] = summary['Total_xG_For'] / summary['Games_Played']
    summary['xG_Against_Per_Game'] = summary['Total_xG_Against'] / summary['Games_Played']

    final_cols = [
        'Adj_xG_For_Per_Game', 'Adj_xG_Against_Per_Game',
        'xG_For_Per_Game', 'xG_Against_Per_Game', 
        'Total_xG_For', 'Total_G_For', 
        'Total_xG_Against', 'Total_G_Against', 'Games_Played'
    ]
    
    summary = summary[final_cols].reset_index().rename(columns={'formation': 'Formation'})
    summary.to_csv(output_filename, index=False)
    print("stats saved.")