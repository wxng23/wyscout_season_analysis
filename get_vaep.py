import os
import ast
import pandas as pd
import tqdm
import warnings
import socceraction.spadl as spadl
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab
import socceraction.vaep.formula as vaepformula
import xgboost
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# --- 1. CONFIGURATION & UTILS ---

def get_features_functions():
    """Centralized list of feature functions to ensure consistency."""
    return [
        fs.actiontype, fs.actiontype_onehot, fs.bodypart, fs.bodypart_onehot, 
        fs.result, fs.result_onehot, fs.goalscore, fs.startlocation,
        fs.endlocation, fs.movement, fs.space_delta, fs.startpolar,
        fs.endpolar, fs.team, fs.time_delta
    ]

def safe_parse(val):
    try:
        return ast.literal_eval(val) if pd.notnull(val) else {}
    except:
        return {}

def dist_to_own_goal(gamestates):
    # start_x is 0-105. Distance to 0 is just start_x.
    a0 = gamestates[0]
    return pd.DataFrame({"dist_to_own_goal": a0.start_x})

# --- 2. DATA PARSING (NCAA -> SPADL) ---

def parse_ncaa_csv(file_path):
    """Parses NCAA Event CSV into standardized SPADL format."""
    df = pd.read_csv(file_path)
    
    def process_row(row):
        type_data = safe_parse(row['type'])
        pass_data = safe_parse(row['pass'])
        shot_data = safe_parse(row['shot'])
        duel_data = safe_parse(row.get('ground_duel', '{}'))
        secondary = type_data.get('secondary', [])
        primary = str(type_data.get('primary', '')).lower()
        
        team_data = safe_parse(row['team'])
        team_id = team_data.get('id', 0)
        loc = safe_parse(row['location'])
        start_x, start_y = loc.get('x', 0) * 1.05, loc.get('y', 0) * 0.68
        
        # TYPE MAPPING
        type_id = 0 
        if row.get('shot') and str(row['shot']) != 'nan': type_id = 11
        elif primary == 'clearance': type_id = 18
        elif primary == 'interception': type_id = 10
        elif primary == 'duel':
            if 'defensive_duel' in secondary:
                type_id = 9
            elif 'offensive_duel' in secondary:
                type_id = 7
            else:
                type_id = 9
        elif primary == 'touch' and 'carry' in secondary: type_id = 21

        # RESULT
        result_id = 0
        
        if primary in ['interception', 'clearance']:
            result_id = 1
        elif type_id == 9: # Defensive Duel
            # Success if: recovered ball, stopped progress, OR the opponent didn't 'win'
            success_indicators = ['recoveredpossession', 'stoppedprogress', 'win', 'success']
            if any(indicator in str(duel_data).lower() for indicator in success_indicators) or type_data.get('success'):
                result_id = 1
            # Neutral outcome check: if the ball went out of play but the defender made the play
            elif 'out' in secondary or 'neutral' in secondary:
                result_id = 1
        elif type_id == 7: # Offensive Duel
            if duel_data.get('progressedWithBall') or duel_data.get('keptPossession'):
                result_id = 1 
        # Standard offensive success (Passes/Shots)
        elif shot_data.get('isGoal') or shot_data.get('onTarget') or pass_data.get('accurate') or type_data.get('success'):
            result_id = 1

        # END LOC
        end_loc = pass_data.get('endLocation') or safe_parse(row.get('carry', {})).get('endLocation') or {}
        if end_loc:
            end_x = end_loc.get('x', 0) * 1.05
            end_y = end_loc.get('y', 0) * 0.68
        else:
            if result_id == 1 and type_id in [9, 10, 18]:
                end_x = min(105.0, start_x + 10.0)
                end_y = start_y
            else:
                end_x = start_x
                end_y = start_y
        bodypart_id = 1 if 'head_pass' in secondary else 0

        return pd.Series([team_id, start_x, start_y, end_x, end_y, type_id, result_id, bodypart_id])

    cols = ['team_id', 'start_x', 'start_y', 'end_x', 'end_y', 'type_id', 'result_id', 'bodypart_id']
    df[cols] = df.apply(process_row, axis=1)
    actions = df.rename(columns={'matchId': 'game_id', 'matchPeriod': 'period_id', 'second': 'time_seconds'})
    actions['period_id'] = actions['period_id'].map({'1H': 1, '2H': 2}).fillna(1)
    
    return actions[['game_id', 'period_id', 'time_seconds', 'team_id', 'start_x', 'start_y', 'end_x', 'end_y', 'type_id', 'result_id', 'bodypart_id']]

# --- 3. FEATURE & LABEL GENERATION ---

def build_vaep_data(csv_path, output_folder):
    """Generates and stores features/labels."""
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    actions = parse_ncaa_csv(csv_path)
    game_id = actions['game_id'].iloc[0]
    actions_named = spadl.add_names(actions)
    
    xfns = get_features_functions()
    gamestates = fs.gamestates(actions_named, 3)
    X = pd.concat([fn(gamestates) for fn in xfns], axis=1)
    
    yfns = [lab.scores, lab.concedes]
    Y = pd.concat([fn(actions_named) for fn in yfns], axis=1)

    for df_tmp in [X, Y]:
        cat_cols = df_tmp.select_dtypes(include=['category']).columns
        df_tmp[cat_cols] = df_tmp[cat_cols].astype(object)

    X.to_hdf(os.path.join(output_folder, "features.h5"), key=f"game_{game_id}", format='table')
    Y.to_hdf(os.path.join(output_folder, "labels.h5"), key=f"game_{game_id}", format='table')

def getXY(game_ids, datafolder, Xcols):
    """Consolidated function to pull all game data for training."""
    features_h5 = os.path.join(datafolder, "features.h5")
    labels_h5 = os.path.join(datafolder, "labels.h5")
    X, Y = [], []
    for g_id in tqdm.tqdm(game_ids, desc="Loading Data"):
        Xi = pd.read_hdf(features_h5, f"game_{g_id}")
        Yi = pd.read_hdf(labels_h5, f"game_{g_id}")
        
        X.append(Xi[Xcols])
        Y.append(Yi)
        
    return pd.concat(X).reset_index(drop=True), pd.concat(Y).reset_index(drop=True)

# --- 4. MODELING ---

def train_vaep_models(X, Y):
    models = {}
    for col in X.select_dtypes(include=['object', 'string']).columns:
        X[col] = X[col].astype('category')
    
    for col in Y.columns:
        print(f"Training: {col}")
        model = xgboost.XGBClassifier(n_estimators=50, max_depth=3, enable_categorical=True)
        model.fit(X, Y[col])
        models[col] = model
    return models

def evaluate_models(models, X, Y):
    for col in Y.columns:
        y_hat = models[col].predict_proba(X)[:, 1]
        print(f"### Eval: {col} | Brier: {brier_score_loss(Y[col], y_hat):.5f} | AUC: {roc_auc_score(Y[col], y_hat):.5f}")

def save_predictions(games, models, Xcols, features_h5, predictions_h5):
    """Generates predictions per game and saves to HDF5 with categorical handling."""
    with pd.HDFStore(predictions_h5) as predictionstore:
        for game_id in tqdm.tqdm(games.game_id, desc="Saving predictions"):
            Xi = pd.read_hdf(features_h5, f"game_{game_id}")[Xcols]
            
            # 1. Convert 'object' columns to 'category' so the model can read them
            obj_cols = Xi.select_dtypes(include=['object', 'string']).columns
            Xi[obj_cols] = Xi[obj_cols].astype('category')
            
            # 2. Generate probabilities
            y_hat = pd.DataFrame({
                col: [p[1] for p in models[col].predict_proba(Xi)]
                for col in models
            })
            
            predictionstore.put(f"game_{int(game_id)}", y_hat)

def compute_player_rankings(data_folder, raw_csv, models, Xcols, output_filename="michigan_player_vaep.csv"):
    raw_df = pd.read_csv(raw_csv)
    actions = parse_ncaa_csv(raw_csv)
    
    # 1. Map player names
    actions['player_name'] = raw_df['player'].apply(
        lambda x: safe_parse(x).get('name', 'Unknown') if pd.notnull(x) else 'Unknown'
    )
    actions_named = spadl.add_names(actions)

    # We must ensure that the groupby keys are simple integers 
    # and categorical columns are reverted to basic types before shifting.
    actions_named['game_id'] = actions_named['game_id'].astype(int)
    actions_named['period_id'] = actions_named['period_id'].astype(int)
    
    for col in actions_named.columns:
        if isinstance(actions_named[col].dtype, pd.CategoricalDtype):
            actions_named[col] = actions_named[col].astype(str)
        if actions_named[col].dtype == object:
            # fillna(0) on the whole DF often fails, so we clean strings here
            actions_named[col] = actions_named[col].fillna("")

    # 3. Generate Gamestates
    xfns = get_features_functions()
    gamestates = fs.gamestates(actions_named, 3)
    
    # 4. Feature Extraction
    X = pd.concat([fn(gamestates) for fn in xfns], axis=1)[Xcols]
    
    # 5. Categorize for XGBoost
    for col in X.select_dtypes(include=['object', 'string']).columns:
        X[col] = X[col].astype('category')

    # 6. Predict & Value
    preds = pd.DataFrame({col: models[col].predict_proba(X)[:, 1] for col in models})
    values = vaepformula.value(actions_named, preds.scores, preds.concedes)

    defensive_mask = (actions_named.type_id.isin([9, 10, 18])) & (actions_named.result_id == 1)
    values.loc[defensive_mask, 'defensive_value'] = values.loc[defensive_mask, 'defensive_value'].clip(lower=-0.005)
    own_half_mask = (actions_named.start_x < 52.5) & defensive_mask
    values.loc[own_half_mask, 'defensive_value'] += 0.02
    values['vaep_value'] = values['offensive_value'] + values['defensive_value']
    
    # 7. Aggregate & Save
    results = pd.concat([actions_named, preds, values], axis=1)
    player_stats = results.groupby("player_name").agg({
        'vaep_value': 'sum',
        'offensive_value': 'sum',
        'defensive_value': 'sum',
        'game_id': 'count'
    }).rename(columns={'game_id': 'action_count'})

    print("\n--- DEFENSIVE AUDIT: TOP 5 'LOWEST VALUE' DEFENSIVE ACTIONS ---")
    # Looking for tackles/interceptions that the model hated
    defensive_actions = results[results['type_name'].isin(['tackle', 'interception', 'clearance'])]
    audit = defensive_actions.sort_values("defensive_value", ascending=True).head(5)
    for _, row in audit.iterrows():
        print(f"Player: {row['player_name']} | Action: {row['type_name']} | "
              f"Loc: ({row['start_x']:.1f}, {row['start_y']:.1f}) | Value: {row['defensive_value']:.4f}")


    # Normalization & Saving
    player_stats['vaep_per_90'] = player_stats['vaep_value'] 
    ranked_stats = player_stats.sort_values("vaep_value", ascending=False)
    
    output_path = os.path.join(data_folder, output_filename)
    ranked_stats.to_csv(output_path)
    
    print(f"Success! Rankings saved to: {output_path}")
    return ranked_stats