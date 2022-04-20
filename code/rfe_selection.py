# # rfe Selection
# from pathlib import path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import typer
import datetime
import os
import shutil

from globals import DATA_MODELLING_FOLDER

def identify_top_features(X, y, estimator, step = 10, n_feat_select = 30):
    ranks = pd.DataFrame(index = X.columns)
    for i in range(10):
        print(i)
        rfecv = RFECV(estimator = RandomForestClassifier(class_weight ='balanced'), step = step , cv = 10, scoring = 'balanced_accuracy', n_jobs = -1)
        rfecv.fit(X, y.to_numpy().ravel())
        print(rfecv.n_features_)
        ranks[str(i)] = rfecv.ranking_
        
    ranks['total_rank'] = np.sum(ranks, axis =1)

    print(ranks.sort_values(by = "total_rank").head(30))

    selected_columns = ranks.sort_values(by = "total_rank").head(30).index

    return selected_columns
    
def main(eval_type : str, rfe_step : int  = typer.Argument(10), n_features :int = typer.Argument(30)):

    startTime = datetime.datetime.now()

    # Assure the type of dataset is correctly established
    assert eval_type in ['full', 'over'], "eval_type (1st argument) was not valid. The only 2 options are 'full', 'over'."

    if eval_type == "full": 
        x_train_data = "X_train_77"
        y_train_data = "y_train_77"
        x_test_data = "X_test_77"
        y_test_data = "y_test_77"


    elif eval_type == "over":
        x_train_data = "X_train_77_over"
        y_train_data = "y_train_77"
        x_test_data = "X_test_77_over"
        y_test_data = "y_test_77"

    # Create pathing
    x_train_path = os.path.join(DATA_MODELLING_FOLDER, x_train_data)
    y_train_path = os.path.join(DATA_MODELLING_FOLDER, y_train_data)
    x_test_path = os.path.join(DATA_MODELLING_FOLDER, x_test_data)
   
    X_train = pd.read_csv(x_train_path + ".csv").set_index('subjectkey')
    y_train = pd.read_csv(y_train_path + ".csv").set_index('subjectkey')
    X_test = pd.read_csv(x_test_path + ".csv").set_index('subjectkey')

    # Run 10 x 10CV RFE procedure and find the avg top n features to be selected (based on RFE Rank)
    selected_cols = identify_top_features(X_train, y_train, RandomForestClassifier(class_weight ='balanced'), rfe_step, n_features)

    # Create new dfs and save
    X_train_rfe = X_train.loc[:, selected_cols]
    X_test_rfe = X_test.loc[:, selected_cols]


    print(X_train_rfe.head())

    X_train_rfe.to_csv(DATA_MODELLING_FOLDER + "/" + x_train_data + "_rfe.csv", index = False)
    print(f"Completed in: {datetime.datetime.now() - startTime}")

if __name__ == "__main__":
    typer.run(main)