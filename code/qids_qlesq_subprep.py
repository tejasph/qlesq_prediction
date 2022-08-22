#ElasticNet_selection
import pandas as pd


import typer
import datetime
import os

from globals import DATA_MODELLING_FOLDER

def main(eval_type : str):

    startTime = datetime.datetime.now()

    # Assure the type of dataset is correctly established
    assert eval_type in ['full', 'over'], "eval_type (1st argument) was not valid. The only 2 options are 'full' or 'over'."

    if eval_type == "full": 
        x_train_data = "X_train_77"
        y_train_data = "y_train_77"
        x_test_data = "X_test_77"
        y_test_data = "y_test_77"

        qids_term = "qids"
        qlesq_term = "qlesq"

    elif eval_type == "over":

        x_train_data = "X_train_77_over"
        y_train_data = "y_train_77"
        x_test_data = "X_test_77_over"
        y_test_data = "y_test_77"

        qids_term = "QIDS"
        qlesq_term = "QLESQ"


    # Create pathing
    x_train_path = os.path.join(DATA_MODELLING_FOLDER, x_train_data)
    y_train_path = os.path.join(DATA_MODELLING_FOLDER, y_train_data)
    x_test_path = os.path.join(DATA_MODELLING_FOLDER, x_test_data)
   
    X_train = pd.read_csv(x_train_path + ".csv").set_index('subjectkey')
    y_train = pd.read_csv(y_train_path + ".csv").set_index('subjectkey')
    X_test = pd.read_csv(x_test_path + ".csv").set_index('subjectkey')


    # Grab Qlesq and QIDS columns
    qids_cols = list(X_train.loc[:, X_train.columns.str.contains(qids_term)].columns)
    qlesq_cols = list(X_train.loc[:, X_train.columns.str.contains(qlesq_term)].columns)

    # Creat qlesq/qids sub dataframes
    X_train_qids = X_train[qids_cols]
    X_test_qids = X_test[qids_cols]

    X_train_qlesq = X_train[qlesq_cols]
    X_test_qlesq = X_test[qlesq_cols]

    X_train_qidsqlesq = X_train[qlesq_cols + qids_cols]
    X_test_qidsqlesq = X_test[qlesq_cols + qids_cols]

    X_train_noqidsqlesq = X_train.drop(columns = qlesq_cols + qids_cols)
    X_test_noqidsqlesq = X_test.drop(columns = qlesq_cols + qids_cols)


    # Export dataframes
    X_train_qids.to_csv(DATA_MODELLING_FOLDER + "/" + x_train_data + "_qids.csv", index = True)
    X_test_qids.to_csv(DATA_MODELLING_FOLDER + "/" + x_test_data + "_qids.csv", index = True)

    X_train_qlesq.to_csv(DATA_MODELLING_FOLDER + "/" + x_train_data + "_qlesq.csv", index = True)
    X_test_qlesq.to_csv(DATA_MODELLING_FOLDER + "/" + x_test_data + "_qlesq.csv", index = True)

    X_train_qidsqlesq.to_csv(DATA_MODELLING_FOLDER + "/" + x_train_data + "_qidsqlesq.csv", index = True)
    X_test_qidsqlesq.to_csv(DATA_MODELLING_FOLDER + "/" + x_test_data + "_qidsqlesq.csv", index = True)

    X_train_noqidsqlesq.to_csv(DATA_MODELLING_FOLDER + "/" + x_train_data + "_noqidsqlesq.csv", index = True)
    X_test_noqidsqlesq.to_csv(DATA_MODELLING_FOLDER + "/" + x_test_data + "_noqidsqlesq.csv", index = True)    

    print(f"Completed in: {datetime.datetime.now() - startTime}")

if __name__ == "__main__":
    typer.run(main)