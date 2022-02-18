# run_experiment

# Basics
import pandas as pd
import numpy as np
import datetime
import typer
import os

# Import paths
from globals import DATA_MODELLING_FOLDER, EXPERIMENT_RESULTS

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

class experiment_manager():
    
    def __init__(self, X, y):
        
        # Establish data, models, and runs for the experiment
        self.X = X
        self.y = y
        self.models = dict()
        self.runs = 10

        
    def CV_run(self, X, y, pipe_model):    
        """
        Does a 10-fold cross validation using a prepared pipeline object

        Returns CV metric results
        """
        # cv done in a stratified fashion to retain class distribution consistency
        cv = StratifiedKFold(n_splits = 10)
        j = 1
        scores = {'fold':[], 
                  'train_bal_acc':[],'train_auc':[],'train_sens':[], 'train_spec': [], 'train_prec':[], 'train_npv':[],
                  'valid_bal_acc':[], 'valid_auc':[], 'valid_sens':[], 'valid_spec':[], 'valid_prec':[], 'valid_npv': []}
        
        # For each cv split...
        for train_id, valid_id in cv.split(X, y.target):
            
            # Subset data based on split indices
            X_train = X.loc[train_id].set_index('subjectkey')
            y_train = y.loc[train_id].set_index('subjectkey')

            X_valid = X.loc[valid_id].set_index('subjectkey')
            y_valid = y.loc[valid_id].set_index('subjectkey')

            # Fit model
            pipe_model.fit(X_train, y_train.target)

            # Collect model predictions
            training_predictions = pipe_model.predict(X_train)
            validation_predictions = pipe_model.predict(X_valid)
            
            # Obtain probabilities for calculation of AUC
            y_train_score = pipe_model.predict_proba(X_train)[:,1]
            y_valid_score = pipe_model.predict_proba(X_valid)[:,1]
            
            # Obtain AUC scores
            train_roc = roc_auc_score(y_train, y_train_score)
            valid_roc = roc_auc_score(y_valid, y_valid_score)

            # Obtain training sensitivity, specificity, PPV/precision, and NPV
            t_tn, t_fp, t_fn, t_tp = confusion_matrix(y_train,training_predictions).ravel()
            train_specificity = t_tn/(t_tn+t_fp)
            train_sensitivity = t_tp/(t_tp+t_fn)
            train_precision = t_tp/(t_tp+t_fp)
            train_npv = t_tn/(t_tn + t_fn)

            # Obtain validation sensitivity, specificity, PPV/precision, and NPV
            v_tn, v_fp, v_fn, v_tp = confusion_matrix(y_valid,validation_predictions).ravel()
            valid_specificity = v_tn/(v_tn + v_fp)
            valid_sensitivity = v_tp/(v_tp + v_fn)
            valid_precision = v_tp/(v_tp + v_fp)
            valid_npv = v_tn/(v_tn + v_fn)
            
            # Store scores for the current split
            scores['fold'].append(j)
            scores['train_bal_acc'].append(balanced_accuracy_score(y_train, training_predictions))
            scores['valid_bal_acc'].append(balanced_accuracy_score(y_valid, validation_predictions))

            scores['train_auc'].append(train_roc)
            scores['valid_auc'].append(valid_roc)

            scores['train_sens'].append(train_sensitivity)
            scores['train_spec'].append(train_specificity)
            scores['train_prec'].append(train_precision)
            scores['train_npv'].append(train_npv)

            scores['valid_sens'].append(valid_sensitivity)
            scores['valid_spec'].append(valid_specificity)
            scores['valid_prec'].append(valid_precision)
            scores['valid_npv'].append(valid_npv)

            j += 1

        # Get avg scores across the 10 splits
        score_df = pd.DataFrame(scores)

        avg_t_bal_acc = score_df.train_bal_acc.mean()
        avg_v_bal_acc = score_df.valid_bal_acc.mean()

        avg_t_auc = score_df.train_auc.mean()
        avg_v_auc = score_df.valid_auc.mean()

        avg_t_sens = score_df.train_sens.mean()
        avg_t_spec = score_df.train_spec.mean()
        avg_t_prec = score_df.train_prec.mean()
        avg_t_npv = score_df.train_npv.mean()
        avg_v_sens = score_df.valid_sens.mean()
        avg_v_spec = score_df.valid_spec.mean()
        avg_v_prec = score_df.valid_prec.mean()
        avg_v_npv = score_df.valid_npv.mean()

        return avg_t_bal_acc, avg_t_auc ,avg_t_sens, avg_t_spec, avg_t_prec, avg_t_npv, avg_v_bal_acc, avg_v_auc, avg_v_sens, avg_v_spec, avg_v_prec, avg_v_npv

    
    def run_experiment(self):
        startTime = datetime.datetime.now()
        
        assert len(self.models) >= 1, "No models to run test for!"
        exp_results = {'model':[], 
       'avg_train_bal_acc':[], 'avg_train_auc':[], 'avg_train_sens':[], 'avg_train_spec':[], 'avg_train_ppv':[], 'avg_train_npv':[],
      'avg_valid_bal_acc':[], 'avg_valid_auc':[], 'avg_valid_sens':[], 'avg_valid_spec':[], 'avg_valid_ppv':[], 'avg_valid_npv':[]}

        std_results = {'model':[], 
       'std_train_bal_acc':[], 'std_train_auc':[], 'std_train_sens':[], 'std_train_spec':[], 'std_train_ppv':[], 'std_train_npv':[],
      'std_valid_bal_acc':[], 'std_valid_auc':[], 'std_valid_sens':[], 'std_valid_spec':[], 'std_valid_ppv':[], 'std_valid_npv':[]}
        
        for model_name, model in self.models.items():
            #print(f"Running {model_name}")
            run = 1
            # Track run results
            runs_dict = {'run':[], 
                      't_bal_acc':[],'t_auc':[],'t_sens':[], 't_spec': [], 't_ppv':[], 't_npv':[],
                      'v_bal_acc':[], 'v_auc':[], 'v_sens':[], 'v_spec':[], 'v_ppv':[], 'v_npv':[]}
            # Run experiment multiple times (ex. 100)
            for r in range(self.runs):


                # Create a pipline with scaling procedure and model of interest
                pipe = Pipeline([('scaler', MinMaxScaler()), model])

                runs_dict['run'].append(run)

                run_results = self.CV_run(self.X, self.y, pipe)
                
                # Unloading tuple in order of: 
                # avg_t_bal_acc, avg_t_auc ,avg_t_sens, avg_t_spec, avg_t_prec, avg_t_npv, (0-5)
                # avg_v_bal_acc, avg_v_auc, avg_v_sens, avg_v_spec, avg_v_prec, avg_v_npv (6-11)
                
                runs_dict['t_bal_acc'].append(run_results[0])
                runs_dict['t_auc'].append(run_results[1])
                runs_dict['t_sens'].append(run_results[2])
                runs_dict['t_spec'].append(run_results[3])
                runs_dict['t_ppv'].append(run_results[4])
                runs_dict['t_npv'].append(run_results[5])

                runs_dict['v_bal_acc'].append(run_results[6])
                runs_dict['v_auc'].append(run_results[7])
                runs_dict['v_sens'].append(run_results[8])
                runs_dict['v_spec'].append(run_results[9])
                runs_dict['v_ppv'].append(run_results[10])
                runs_dict['v_npv'].append(run_results[11])

                run +=1

            runs_df = pd.DataFrame(runs_dict)

            # Calculate avg scores across all runs
            exp_results['model'].append(model_name)
            exp_results['avg_train_bal_acc'].append(runs_df['t_bal_acc'].mean())
            exp_results['avg_train_auc'].append(runs_df['t_auc'].mean())
            exp_results['avg_train_sens'].append(runs_df['t_sens'].mean())
            exp_results['avg_train_spec'].append(runs_df['t_spec'].mean())
            exp_results['avg_train_ppv'].append(runs_df['t_ppv'].mean())
            exp_results['avg_train_npv'].append(runs_df['t_npv'].mean())

            exp_results['avg_valid_bal_acc'].append(runs_df['v_bal_acc'].mean())
            exp_results['avg_valid_auc'].append(runs_df['v_auc'].mean())
            exp_results['avg_valid_sens'].append(runs_df['v_sens'].mean())
            exp_results['avg_valid_spec'].append(runs_df['v_spec'].mean())
            exp_results['avg_valid_ppv'].append(runs_df['v_ppv'].mean())
            exp_results['avg_valid_npv'].append(runs_df['v_npv'].mean())

            std_results['model'].append(model_name)
            std_results['std_train_bal_acc'].append(runs_df['t_bal_acc'].std())
            std_results['std_train_auc'].append(runs_df['t_auc'].std())
            std_results['std_train_sens'].append(runs_df['t_sens'].std())
            std_results['std_train_spec'].append(runs_df['t_spec'].std())
            std_results['std_train_ppv'].append(runs_df['t_ppv'].std())
            std_results['std_train_npv'].append(runs_df['t_npv'].std())

            std_results['std_valid_bal_acc'].append(runs_df['v_bal_acc'].std())
            std_results['std_valid_auc'].append(runs_df['v_auc'].std())
            std_results['std_valid_sens'].append(runs_df['v_sens'].std())
            std_results['std_valid_spec'].append(runs_df['v_spec'].std())
            std_results['std_valid_ppv'].append(runs_df['v_ppv'].std())
            std_results['std_valid_npv'].append(runs_df['v_npv'].std())

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



def main(x_data: str, y_data: str):

    startTime = datetime.datetime.now()

    x_path = os.path.join(DATA_MODELLING_FOLDER, x_data)
    y_path = os.path.join(DATA_MODELLING_FOLDER, y_data)
   
    X = pd.read_csv(x_path + ".csv")
    y = pd.read_csv(y_path + ".csv")


    y.columns = ['subjectkey', 'target']

    exp_1 = experiment_manager(X, y)

    exp_1.models = {'Dummy Classification': ('dummy', DummyClassifier(strategy = 'stratified')), 
          'Random Forest' : ('rf', RandomForestClassifier(criterion = 'entropy', class_weight = 'balanced', max_depth = 78, max_features = 'log2', min_impurity_decrease = 0.1, min_samples_leaf = 5, min_samples_split = 2)),
          'Logistic Regression': ('lr', LogisticRegression(solver = 'liblinear', class_weight = 'balanced', penalty = 'l1', C = 0.099, tol = 0.1)),
          'KNearest Neighbors': ('knn', KNeighborsClassifier(n_neighbors = 15, p = 1, weights = 'uniform')),
          'Support Vector Machine':('svc', SVC(class_weight = 'balanced', C = 1, gamma = 'auto', probability = True) )}

    exp_1.run_experiment()

    print(exp_1.avg_results)
    print(exp_1.std_results)

    exp_1.store_results(EXPERIMENT_RESULTS, "exp_1_CV_full_features")


if __name__ == "__main__":
    typer.run(main)