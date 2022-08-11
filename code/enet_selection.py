#ElasticNet_selection
import pandas as pd
from sklearn.linear_model import ElasticNetCV

import typer
import datetime
import os

from globals import DATA_MODELLING_FOLDER

def main(eval_type : str):

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

    enet_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9], eps=0.005, n_alphas=30, fit_intercept=True,
                    normalize=True, precompute='auto', max_iter=1000, tol=0.0001, cv=5,
                    copy_X=True, n_jobs=-1)

    enet_model.fit(X_train, y_train.to_numpy().ravel())

    print(f"Best alpha is {enet_model.alpha_} and best l1_ratio is  {enet_model.l1_ratio_}")

    coef_df = pd.DataFrame({'coef':enet_model.coef_}, index = X_train.columns)

    selected_cols = coef_df[coef_df['coef'] != 0].index
    
    print(f"The number of selected features after cross-validation is {len(selected_cols)}")

    # create feature selected dfs for export
    X_train_enet = X_train.loc[:, selected_cols]
    X_test_enet = X_test.loc[:, selected_cols]


    X_train_enet.to_csv(DATA_MODELLING_FOLDER + "/" + x_train_data + "_enet.csv", index = True)
    X_test_enet.to_csv(DATA_MODELLING_FOLDER + "/" + x_test_data + "_enet.csv", index = True)

    print(f"Completed in: {datetime.datetime.now() - startTime}")

if __name__ == "__main__":
    typer.run(main)