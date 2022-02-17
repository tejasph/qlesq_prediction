
#gridsearch module.py
import os
import typer
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

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

def get_scaled_pipeline(model):
    return Pipeline([('scaler', MinMaxScaler()), model])
        

def main(x_data: str, y_data: str):

    startTime = datetime.datetime.now()

    x_path = os.path.join(DATA_MODELLING_FOLDER, x_data)
    y_path = os.path.join(DATA_MODELLING_FOLDER, y_data)
   
    X = pd.read_csv(x_path + ".csv").set_index('subjectkey')
    y = pd.read_csv(y_path + ".csv").set_index('subjectkey')


    y.columns = ['target']

###### RF grid search
    # rf =('rf', RandomForestClassifier(n_estimators = 100, class_weight = 'balanced')) 
    # rf_params = {'rf__max_features': ['sqrt', 'log2', 0.33, 0.2, 0.1],
    #         'rf__max_depth': [int(x) for x in np.linspace(2, 100, num = 10)],
    #         'rf__min_samples_split': [2,4,6,8,10],
    #         'rf__min_samples_leaf': [1,2,3,4,5],
    #         'rf__min_impurity_decrease': [0.0, 0.1, 0.3],
    #         'rf__criterion':['gini', 'entropy']}

    # rf_pipe = get_scaled_pipeline(rf)

    # optimizer = model_optimizer(rf_pipe, rf_params, X, y, "rf_full_broad_grid")

    # optimizer.search_grid()

    # optimizer.write_results()

######

###### Logistic Regression Grid Search

    # lr = ('lr', LogisticRegression(penalty = 'l2', class_weight = 'balanced', solver = 'liblinear'))
    # lr_params = {'lr__tol' : [0.1, 0.01, 0.001, 'none'],
    #             'lr__C': [p/1000 for p in range(90, 120, 1)],
    #             'lr__penalty': ['l1', 'l2']}

    # lr_pipe = get_scaled_pipeline(lr)

    # lr_optimizer = model_optimizer(lr_pipe, lr_params, X, y, "lr_full_broad_grid")

    # lr_optimizer.search_grid()

    # lr_optimizer.write_results()

###### 

###### KNeighbors Classifier

    knn = ('knn', KNeighborsClassifier())

    knn_params = {'knn__n_neighbors': [n for n in range(1,31, 2)],
                'knn__weights': ['uniform', 'distance'], 
                'knn__p': [1,2]}

    knn_pipe = get_scaled_pipeline(knn)

    knn_optimizer = model_optimizer(knn_pipe, knn_params, X, y, "knn_full_broad_grid")

    knn_optimizer.search_grid()
    knn_optimizer.write_results()

    print(f"Completed in: {datetime.datetime.now() - startTime}")


if __name__ == "__main__":
    typer.run(main)


