import get_data
import get_xG
import get_vaep
import os
import ast
import tqdm
import warnings
import pandas as pd
import socceraction.spadl as spadl
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab
from sklearn.metrics import brier_score_loss, roc_auc_score

def main():
# --- DATA ACQUISITION SECTION ---
    # Uncomment the lines below to pull fresh data from the API
    seasonId = get_data.get_current_big_ten_season()
    get_data.getSeason(seasonId)
    get_data.getUmichGame(seasonId)
    get_data.getUmichOnly(seasonId)

    # --- xG PROCESSING SECTION ---
    # Uncomment to process baseline xG stats before running VAEP
    # get_xG.process_regular_season_xg("seasonEvents25.csv")
    # get_xG.process_formation_stats("umichGameEvents25.csv", "big10_xg.csv")

    # --- VAEP PROCESSING SECTION ---

    # --- FILE PATHS ---
    # big10_raw = "seasonEvents25.csv"
    # umich_raw = "umichOnlyEvents25.csv"
    # data_folder = "./vaep_data"
    
    # # 1. CLEANUP SECTION
    # # We clear the HDF5 files to ensure the model doesn't "hallucinate" old data
    # files_to_clean = [
    #     os.path.join(data_folder, "features.h5"),
    #     os.path.join(data_folder, "labels.h5"),
    #     os.path.join(data_folder, "predictions.h5"),
    #     os.path.join(data_folder, "michigan_player_vaep.csv")
    # ]
    # for f in files_to_clean:
    #     if os.path.exists(f):
    #         os.remove(f)
    #         print(f"Cleaned old file: {f}")

    # # 2. DATA PRE-PROCESSING (Big Ten)
    # print("Step 1: Parsing Big Ten season data and building feature stores...")
    # # This creates the training set from the entire league
    # get_vaep.build_vaep_data(big10_raw, data_folder)

    # # We need the game_id from the Big Ten file to pull it from the HDF5 store
    # big10_actions = get_vaep.parse_ncaa_csv(big10_raw)
    # big10_game_id = big10_actions['game_id'].iloc[0]

    # # 3. FEATURE DEFINITION
    # # Centralizing this list ensures consistency between Big Ten and Michigan processing
    # xfns = get_vaep.get_features_functions()
    # Xcols = fs.feature_column_names(xfns, nb_prev_actions=1)

    # # 4. MODEL TRAINING (The "Brain")
    # print("Step 2: Training VAEP models on league-wide data...")
    # # We pass [big10_game_id] as a list to match getXY's signature
    # X_train, Y_train = get_vaep.getXY([big10_game_id], data_folder, Xcols)
    # models = get_vaep.train_vaep_models(X_train, Y_train)

    # # 5. EVALUATION
    # print("Step 3: Evaluating model accuracy (Big Ten performance)...")
    # get_vaep.evaluate_models(models, X_train, Y_train)

    # # 6. PLAYER RANKINGS (The "Report")
    # print("Step 4: Applying Big Ten intelligence to Michigan players...")
    # # We pass the trained models and Xcols directly to the ranking function
    # rankings = get_vaep.compute_player_rankings(data_folder, umich_raw, models, Xcols)
    
    # print("\n" + "="*40)
    # print("MICHIGAN SEASON VAEP LEADERS")
    # print("="*40)
    # print(rankings.head(15))
    # print("="*40)
    # print(f"Full report saved in {data_folder}/michigan_player_vaep.csv")
    
if __name__ == "__main__":
    main()