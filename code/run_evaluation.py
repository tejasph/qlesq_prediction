# run_experiment

# Basics
import pandas as pd
import numpy as np
import datetime
import typer
import os

# Import paths
from globals import DATA_MODELLING_FOLDER, EVALUATION_RESULTS, full_feat_models, overlapping_feat_models

# Import sklearn processing/pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

class evaluation_manager():
    
    def __init__(self, X_train,y_train, X_test, y_test, x_train_data, y_train_data, x_test_data, y_test_data):
        
        # Establish data, models, and runs for the experiment
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_train_type = x_train_data
        self.y_train_type = y_train_data
        self.X_test_type = x_test_data
        self.y_test_type = y_test_data
        self.models = dict()
        self.runs = 10

        
    def eval_run(self, pipe_model):    
        """
        Evaluates prepare pipeline model

        Returns run results
        """

        # Fit model
        pipe_model.fit(self.X_train, self.y_train.target)

        # Collect model predictions
        training_predictions = pipe_model.predict(self.X_train)
        test_predictions = pipe_model.predict(self.X_test)

        # Obtain probabilities for calculation of AUC
        y_train_score = pipe_model.predict_proba(self.X_train)[:,1]
        y_test_score = pipe_model.predict_proba(self.X_test)[:,1]

        # Obtain AUC scores
        train_auc = roc_auc_score(self.y_train, y_train_score)
        test_auc = roc_auc_score(self.y_test, y_test_score)

        # Obtain training sensitivity, specificity, PPV/precision, and NPV
        train_tn, train_fp, train_fn, train_tp = confusion_matrix(self.y_train, training_predictions).ravel()
        train_spec = train_tn/(train_tn + train_fp)
        train_sens = train_tp/(train_tp + train_fn)
        train_prec = train_tp/(train_tp + train_fp)
        train_npv = train_tn/(train_tn + train_fn)

        # Obtain validation sensitivity, specificity, PPV/precision, and NPV
        test_tn, test_fp, test_fn, test_tp = confusion_matrix(self.y_test, test_predictions).ravel()
        test_spec = test_tn/(test_tn + test_fp)
        test_sens = test_tp/(test_tp + test_fn)
        test_prec = test_tp/(test_tp + test_fp)
        test_npv = test_tn/(test_tn + test_fn)

        # Store scores for the current split
        train_bal_acc = balanced_accuracy_score(self.y_train, training_predictions)
        test_bal_acc = balanced_accuracy_score(self.y_test, test_predictions)

        return train_bal_acc, train_auc, train_sens, train_spec, train_prec, train_npv, test_bal_acc, test_auc, test_sens, test_spec, test_prec, test_npv

    
    def run_evaluation(self):
        startTime = datetime.datetime.now()
        
        assert len(self.models) >= 1, "No models to run test for!"
        exp_results = {'model':[], 
       'avg_train_bal_acc':[], 'avg_train_auc':[], 'avg_train_sens':[], 'avg_train_spec':[], 'avg_train_ppv':[], 'avg_train_npv':[],
      'avg_test_bal_acc':[], 'avg_test_auc':[], 'avg_test_sens':[], 'avg_test_spec':[], 'avg_test_ppv':[], 'avg_test_npv':[]}

        std_results = {'model':[], 
       'std_train_bal_acc':[], 'std_train_auc':[], 'std_train_sens':[], 'std_train_spec':[], 'std_train_ppv':[], 'std_train_npv':[],
      'std_test_bal_acc':[], 'std_test_auc':[], 'std_test_sens':[], 'std_test_spec':[], 'std_test_ppv':[], 'std_test_npv':[]}
        
        for model_name, model in self.models.items():
            #print(f"Running {model_name}")
            run = 1
            # Track run results
            runs_dict = {'run':[], 
                      'train_bal_acc':[],'train_auc':[],'train_sens':[], 'train_spec': [], 'train_ppv':[], 'train_npv':[],
                      'test_bal_acc':[], 'test_auc':[], 'test_sens':[], 'test_spec':[], 'test_ppv':[], 'test_npv':[]}
            # Run experiment multiple times (ex. 100)
            for r in range(self.runs):

                # Create a pipline with scaling procedure and model of interest
                pipe = Pipeline([('scaler', MinMaxScaler()), model])

                runs_dict['run'].append(run)

                run_results = self.eval_run(pipe)
                
                # Unloading tuple in order of: 
                # avg_t_bal_acc, avg_t_auc ,avg_t_sens, avg_t_spec, avg_t_prec, avg_t_npv, (0-5)
                # avg_v_bal_acc, avg_v_auc, avg_v_sens, avg_v_spec, avg_v_prec, avg_v_npv (6-11)
                
                runs_dict['train_bal_acc'].append(run_results[0])
                runs_dict['train_auc'].append(run_results[1])
                runs_dict['train_sens'].append(run_results[2])
                runs_dict['train_spec'].append(run_results[3])
                runs_dict['train_ppv'].append(run_results[4])
                runs_dict['train_npv'].append(run_results[5])

                runs_dict['test_bal_acc'].append(run_results[6])
                runs_dict['test_auc'].append(run_results[7])
                runs_dict['test_sens'].append(run_results[8])
                runs_dict['test_spec'].append(run_results[9])
                runs_dict['test_ppv'].append(run_results[10])
                runs_dict['test_npv'].append(run_results[11])

                run +=1

            runs_df = pd.DataFrame(runs_dict)

            # Calculate avg scores across all runs
            exp_results['model'].append(model_name)
            exp_results['avg_train_bal_acc'].append(runs_df['train_bal_acc'].mean())
            exp_results['avg_train_auc'].append(runs_df['train_auc'].mean())
            exp_results['avg_train_sens'].append(runs_df['train_sens'].mean())
            exp_results['avg_train_spec'].append(runs_df['train_spec'].mean())
            exp_results['avg_train_ppv'].append(runs_df['train_ppv'].mean())
            exp_results['avg_train_npv'].append(runs_df['train_npv'].mean())

            exp_results['avg_test_bal_acc'].append(runs_df['test_bal_acc'].mean())
            exp_results['avg_test_auc'].append(runs_df['test_auc'].mean())
            exp_results['avg_test_sens'].append(runs_df['test_sens'].mean())
            exp_results['avg_test_spec'].append(runs_df['test_spec'].mean())
            exp_results['avg_test_ppv'].append(runs_df['test_ppv'].mean())
            exp_results['avg_test_npv'].append(runs_df['test_npv'].mean())

            std_results['model'].append(model_name)
            std_results['std_train_bal_acc'].append(runs_df['train_bal_acc'].std())
            std_results['std_train_auc'].append(runs_df['train_auc'].std())
            std_results['std_train_sens'].append(runs_df['train_sens'].std())
            std_results['std_train_spec'].append(runs_df['train_spec'].std())
            std_results['std_train_ppv'].append(runs_df['train_ppv'].std())
            std_results['std_train_npv'].append(runs_df['train_npv'].std())

            std_results['std_test_bal_acc'].append(runs_df['test_bal_acc'].std())
            std_results['std_test_auc'].append(runs_df['test_auc'].std())
            std_results['std_test_sens'].append(runs_df['test_sens'].std())
            std_results['std_test_spec'].append(runs_df['test_spec'].std())
            std_results['std_test_ppv'].append(runs_df['test_ppv'].std())
            std_results['std_test_npv'].append(runs_df['test_npv'].std())

        self.avg_results = pd.DataFrame(exp_results)
        self.std_results = pd.DataFrame(std_results)

        print(f"Completed in: {datetime.datetime.now() - startTime}")

    def store_results(self, results_path, exp_name):
        
        out_path = os.path.join(results_path, exp_name) 

        if os.path.isdir(out_path):
            raise Exception("Name already exists")
        else:
            os.mkdir(out_path + "/")
        
        merged_df = self.avg_results.merge(self.std_results, on = 'model')

        self.avg_results.to_csv(out_path + "/" + exp_name + "_avg.csv" ,index = False)
        self.std_results.to_csv(out_path + "/" + exp_name + "_std.csv" ,index = False)
        merged_df.to_csv(out_path + "/" + exp_name + "_merged.csv" ,index = False)

        f = open(os.path.join(out_path, exp_name + '.txt'), 'w')
        f.write(f"Model used: {self.models}\n\n")
        f.write(f"Trained on  {self.X_train_type} and {self.y_train_type}\n\n")
        f.write(f"Evaluated on  {self.X_test_type} and {self.y_test_type}")


def main(eval_type : str, eval_name : str):

    startTime = datetime.datetime.now()

    if eval_type == "full": 
        x_train_data = "X_train_77"
        y_train_data = "y_train_77"
        x_test_data = "X_test_77"
        y_test_data = "y_test_77"

        models = full_feat_models

    elif eval_type == "over":
        x_train_data = "X_train_77_over"
        y_train_data = "y_train_77"
        x_test_data = "X_test_77_over"
        y_test_data = "y_test_77"

        models = overlapping_feat_models

    x_train_path = os.path.join(DATA_MODELLING_FOLDER, x_train_data)
    y_train_path = os.path.join(DATA_MODELLING_FOLDER, y_train_data)
    x_test_path = os.path.join(DATA_MODELLING_FOLDER, x_test_data)
    y_test_path = os.path.join(DATA_MODELLING_FOLDER, y_test_data)
   
    X_train = pd.read_csv(x_train_path + ".csv").set_index('subjectkey')
    y_train = pd.read_csv(y_train_path + ".csv").set_index('subjectkey')

    X_test = pd.read_csv(x_test_path + ".csv").set_index('subjectkey')
    y_test = pd.read_csv(y_test_path + ".csv").set_index('subjectkey')


    y_train.columns = ['target']
    y_test.columns = ['target']

    eval_1 = evaluation_manager(X_train, y_train, X_test, y_test, x_train_data, y_train_data, x_test_data, y_test_data)

    # Regular Models
    # eval_1.models = {'Dummy Classification': ('dummy', DummyClassifier(strategy = 'stratified')), 
    #       'Random Forest' : ('rf', RandomForestClassifier(class_weight = 'balanced', max_depth = 2, max_features = 'sqrt')),
    #       'Logistic Regression': ('lr', LogisticRegression(solver = 'saga', class_weight = 'balanced', penalty = 'elasticnet', max_iter = 1000,  C = 0.096, tol = 0.1, l1_ratio = 0.8)),
    #       'KNearest Neighbors': ('knn', KNeighborsClassifier(n_neighbors = 15, p = 1, weights = 'uniform')),
    #       'Support Vector Machine':('svc', SVC(class_weight = 'balanced', C = 1, gamma = 'auto', probability = True))}

    # Overlapping models
    # eval_1.models = {'Dummy Classification': ('dummy', DummyClassifier(strategy = 'stratified')),
    #                 'Logistic Regression': ('lr', LogisticRegression(solver = 'saga', class_weight = 'balanced', penalty = 'elasticnet', max_iter = 1000,  C = 0.091, tol = 0.1, l1_ratio = 0.9)),
    #                 'Random Forest' :('rf', RandomForestClassifier(class_weight = 'balanced', max_depth = 2, max_features = 'sqrt')),
    #                 'KNearest Neighbors' :('knn', KNeighborsClassifier(n_neighbors = 1, p = 1, weights = 'uniform')),
    #                 'SVC' :('svc', SVC(class_weight = 'balanced', C= 1, gamma= 'scale', probability = True))}
    eval_1.models = models

    eval_1.run_evaluation()

    print(eval_1.avg_results)
    print(eval_1.std_results)

    eval_1.store_results(EVALUATION_RESULTS, eval_name)


if __name__ == "__main__":
    typer.run(main)