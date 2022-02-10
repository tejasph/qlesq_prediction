
#gridsearch module.py
import os
import typer
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from globals import DATA_MODELLING_FOLDER

from sklearn.preprocessing import MinMaxScaler

class model_optimizer():
    
    def __init__(self, pipeline, params, X, y, name):
        self.pipeline = pipeline
        self.params = params
        self.X = X
        self.y = y
        self.name = name
        
    def search_grid(self, metric = 'balanced_accuracy'):
        self.metric = metric
        self.grid = GridSearchCV(estimator=self.pipeline, n_jobs= -1, scoring = metric, cv = 10, param_grid = self.params, return_train_score = True)
        self.grid.fit(self.X,self.y.target)
        print(f"Best estimator was {self.grid.best_estimator_}")
        print(f"Avg validation score for the estimator was {self.grid.best_score_}")
        
    def write_results(self):
        print(f"Best estimator was {self.grid.best_estimator_}")
        print(f"Avg validation score for the estimator was {self.grid.best_score_}")
        f = open(os.path.join(r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\grid_search", self.name + '.txt'), 'w')
        f.write(f"Model used: {self.pipeline}\n\n")
        f.write(f"Parameter Grid: {self.params}\n\n")
        f.write(f"Best {self.metric} score was: {self.grid.best_score_} \n using the following params: {self.grid.best_params_}")

def create_scaled_pipeline(model):
    return Pipeline([('scaler', MinMaxScaler()), model])
        

def main(x_path: str, y_path: str):

    X = pd.read_csv(x_path).set_index('subjectkey')
    y = pd.read_csv(y_path).set_index('subjectkey')

    y.columns = ['target']

    rf =('rf', RandomForestClassifier()) 
    rf_params = {'rf__max_features': ['sqrt', 'log2', 0.33, 0.2],
            'rf__max_depth': [int(x) for x in np.linspace(2, 100, num = 20)]}

    rf_pipe = create_scaled_pipeline(rf)

    optimizer = model_optimizer(rf_pipe, rf_params, X, y, "rf_script_test")

    optimizer.search_grid()

    optimizer.write_results()


if __name__ == "__main__":
    typer.run(main)


