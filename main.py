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
    print(seasonId)
    # get_data.getSeason(seasonId)
    # get_data.getUmichGame(seasonId)
    # get_data.getUmichOnly(seasonId)

    # --- xG PROCESSING SECTION ---
    # Uncomment to process baseline xG stats before running VAEP
    # get_xG.process_regular_season_xg("seasonEvents25.csv")
    # get_xG.process_formation_stats("umichGameEvents25.csv", "big10_xg.csv")
    
if __name__ == "__main__":
    main()