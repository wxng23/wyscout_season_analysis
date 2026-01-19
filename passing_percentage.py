import pandas as pd
import ast

# Helper function to parse dictionary strings safely
def safe_eval(val):
    try:
        return ast.literal_eval(val) if isinstance(val, str) else val
    except (ValueError, SyntaxError):
        return {}

# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv('umichGameEvents25.csv')

# 2. CLEANING & PREPARATION
# ---------------------------------------------------------
df['team_name'] = df['team'].apply(lambda x: safe_eval(x).get('name') if isinstance(x, str) else x)
df['opponent_name'] = df['opponentTeam'].apply(lambda x: safe_eval(x).get('name') if isinstance(x, str) else x)
df['location'] = df['location'].apply(lambda x: safe_eval(x) if isinstance(x, str) else x)
df['x'] = df['location'].apply(lambda x: x.get('x') if isinstance(x, dict) else None)

# Create field zones
df['field_zone'] = df['x'].apply(lambda x: 
    'Defensive Third' if x is not None and x < 33 else 
    'Middle Third' if x is not None and x < 67 else 
    'Attacking Third' if x is not None else None
)

# 2. FILTER: MICHIGAN PASSES ONLY
# ---------------------------------------------------------
# We only care about Michigan's passes for this specific metric
df['event_primary'] = df['type'].apply(lambda x: safe_eval(x).get('primary') if isinstance(x, str) else x)
mich_passes = df[
    (df['event_primary'] == 'pass') & 
    (df['team_name'] == 'Michigan Wolverines')
].copy()

game_by_game = []

# 3. ANALYSIS LOOP
# ---------------------------------------------------------
for match_id in mich_passes['matchId'].unique():
    match_data = mich_passes[mich_passes['matchId'] == match_id]
    
    # Get Opponent Name (grab from the first row)
    opponent = match_data['opponent_name'].iloc[0]
    
    # Total Michigan Passes
    total_passes = len(match_data)
    
    if total_passes == 0:
        continue
        
    # Count passes in each zone
    def_count = len(match_data[match_data['field_zone'] == 'Defensive Third'])
    mid_count = len(match_data[match_data['field_zone'] == 'Middle Third'])
    att_count = len(match_data[match_data['field_zone'] == 'Attacking Third'])
    
    # Calculate Percentages (Distribution)
    # This answers: "What % of OUR passes were in this zone?"
    pct_def = (def_count / total_passes) * 100
    pct_mid = (mid_count / total_passes) * 100
    pct_att = (att_count / total_passes) * 100
    
    game_by_game.append({
        'Opponent': opponent,
        'Total Passes': total_passes,
        'Def Dist %': pct_def,
        'Mid Dist %': pct_mid,
        'Att Dist %': pct_att
    })

# 4. OUTPUT
# ---------------------------------------------------------
game_df = pd.DataFrame(game_by_game)
pd.options.display.float_format = '{:.1f}'.format

print("MICHIGAN PASSING DISTRIBUTION - GAME-BY-GAME BREAKDOWN")
print("=" * 100)
print(game_df.to_string(index=False))
print(f"\nTotal Games Analyzed: {len(game_df)}")