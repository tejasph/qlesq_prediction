#train_test_split

import pandas as pd
import typer
import datetime
from globals import DATA_MODELLING_FOLDER
from sklearn.model_selection import train_test_split
import os



def main(x_data: str, y_data: str , y_type: str = 'qlesq_QoL_threshold'):
    startTime = datetime.datetime.now()
    typer.echo(x_data)
    typer.echo(y_data)

    X_path = os.path.join(DATA_MODELLING_FOLDER, x_data)
    y_path = os.path.join(DATA_MODELLING_FOLDER, y_data)

    X = pd.read_csv(X_path).set_index('subjectkey')
    y = pd.read_csv(y_path).set_index('subjectkey')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y[[y_type]], stratify = y[[y_type]], test_size=0.20, random_state = 92)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    print(y_train.value_counts())
    print(y_test.value_counts())

    if "77" in x_data:
        if "extval" in x_data:
            name_addon = "77_over"
        else:
            name_addon = "77"
    elif "91" in x_data:
        if "extval" in x_data:
            name_addon = "91_over"
        else:
            name_addon = "91"


    X_train.to_csv(DATA_MODELLING_FOLDER + "/X_train_"+ name_addon +".csv" , index = True)
    y_train.to_csv(DATA_MODELLING_FOLDER + "/y_train_" + name_addon + ".csv", index = True)
    X_test.to_csv(DATA_MODELLING_FOLDER + "/X_test_"+ name_addon + ".csv", index = True)
    y_test.to_csv(DATA_MODELLING_FOLDER + "/y_test_" + name_addon + ".csv", index = True)


    print(f"Completed in: {datetime.datetime.now() - startTime}")



if __name__ == "__main__":
    typer.run(main)