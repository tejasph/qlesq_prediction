# run_experiment

# Basics
import pandas as pd
import numpy as np
import datetime
import typer
import os

# Import paths
from globals import DATA_MODELLING_FOLDER, EXPERIMENT_RESULTS
from globals import full_feat_models, full_enet_feat_models, overlapping_feat_models, overlapping_enet_feat_models
from globals import qids_models, qlesq_models, qidsqlesq_models, noqidsqlesq_models, noqidsqlesq_enet_models
from globals import over_qids_models, over_qlesq_models, over_qidsqlesq_models, over_noqidsqlesq_models
# from globals import over_qids_models, over_qlesq_models, over_qidsqlesq_models, over_noqidsqlesq_models

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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

class experiment_manager():
    
    def __init__(self, X, y, x_data, y_data):
        
        # Establish data, models, and runs for the experiment
        self.X = X
        self.y = y
        self.X_type = x_data
        self.y_type = y_data
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
                  'train_bal_acc':[],'train_acc':[], 'train_auc':[],'train_tp':[], 'train_tn':[], 'train_fp':[], 'train_fn':[], 'train_sens':[], 'train_spec': [], 'train_prec':[], 'train_npv':[], 'train_f1':[],
                  'valid_bal_acc':[],'valid_acc':[], 'valid_auc':[], 'valid_tp':[], 'valid_tn':[], 'valid_fp':[], 'valid_fn':[],  'valid_sens':[], 'valid_spec':[], 'valid_prec':[], 'valid_npv': [], 'valid_f1':[]}
        
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
            train_f1 = f1_score(y_train, training_predictions)

            # Obtain validation sensitivity, specificity, PPV/precision, and NPV
            v_tn, v_fp, v_fn, v_tp = confusion_matrix(y_valid,validation_predictions).ravel()
            valid_specificity = v_tn/(v_tn + v_fp)
            valid_sensitivity = v_tp/(v_tp + v_fn)
            valid_precision = v_tp/(v_tp + v_fp)
            valid_npv = v_tn/(v_tn + v_fn)
            valid_f1 = f1_score(y_valid, validation_predictions)
            
            # Store scores for the current split
            scores['fold'].append(j)
            scores['train_bal_acc'].append(balanced_accuracy_score(y_train, training_predictions))
            scores['train_acc'].append(accuracy_score(y_train, training_predictions))

            scores['valid_bal_acc'].append(balanced_accuracy_score(y_valid, validation_predictions))
            scores['valid_acc'].append(accuracy_score(y_valid, validation_predictions))

            scores['train_auc'].append(train_roc)
            scores['valid_auc'].append(valid_roc)

            scores['train_tp'].append(t_tp)
            scores['train_tn'].append(t_tn)
            scores['train_fp'].append(t_fp)
            scores['train_fn'].append(t_fn)

            scores['valid_tp'].append(v_tp)
            scores['valid_tn'].append(v_tn)
            scores['valid_fp'].append(v_fp)
            scores['valid_fn'].append(v_fn)

            scores['train_sens'].append(train_sensitivity)
            scores['train_spec'].append(train_specificity)
            scores['train_prec'].append(train_precision)
            scores['train_npv'].append(train_npv)
            scores['train_f1'].append(train_f1)

            scores['valid_sens'].append(valid_sensitivity)
            scores['valid_spec'].append(valid_specificity)
            scores['valid_prec'].append(valid_precision)
            scores['valid_npv'].append(valid_npv)
            scores['valid_f1'].append(valid_f1)

            j += 1

        # Get avg scores across the 10 splits
        score_df = pd.DataFrame(scores)

        avg_t_bal_acc = score_df.train_bal_acc.mean()
        avg_t_acc = score_df.train_acc.mean()

        avg_v_bal_acc = score_df.valid_bal_acc.mean()
        avg_v_acc = score_df.valid_acc.mean()

        avg_t_auc = score_df.train_auc.mean()
        avg_v_auc = score_df.valid_auc.mean()

        avg_t_tp = score_df.train_tp.mean()
        avg_t_tn = score_df.train_tn.mean()
        avg_t_fp = score_df.train_fp.mean()
        avg_t_fn = score_df.train_fn.mean()
        avg_t_sens = score_df.train_sens.mean()
        avg_t_spec = score_df.train_spec.mean()
        avg_t_prec = score_df.train_prec.mean()
        avg_t_npv = score_df.train_npv.mean()
        avg_t_f1 = score_df.train_f1.mean()

        avg_v_tp = score_df.valid_tp.mean()
        avg_v_tn = score_df.valid_tn.mean()
        avg_v_fp = score_df.valid_fp.mean()
        avg_v_fn = score_df.valid_fn.mean()
        avg_v_sens = score_df.valid_sens.mean()
        avg_v_spec = score_df.valid_spec.mean()
        avg_v_prec = score_df.valid_prec.mean()
        avg_v_npv = score_df.valid_npv.mean()
        avg_v_f1 = score_df.valid_f1.mean()

        return avg_t_bal_acc, avg_t_acc, avg_t_auc, avg_t_tp, avg_t_tn, avg_t_fp, avg_t_fn, avg_t_sens, avg_t_spec, avg_t_prec, avg_t_npv, avg_t_f1,  avg_v_bal_acc, avg_v_acc, avg_v_auc, avg_v_tp, avg_v_tn, avg_v_fp, avg_v_fn, avg_v_sens, avg_v_spec, avg_v_prec, avg_v_npv, avg_v_f1

    
    def run_experiment(self):
        startTime = datetime.datetime.now()
        
        assert len(self.models) >= 1, "No models to run test for!"
        exp_results = {'model':[], 
       'avg_train_bal_acc':[],'avg_train_acc':[],  'avg_train_auc':[], 'avg_train_tp':[], 'avg_train_tn':[], 'avg_train_fp':[], 'avg_train_fn':[], 'avg_train_sens':[], 'avg_train_spec':[], 'avg_train_ppv':[], 'avg_train_npv':[], 'avg_train_f1':[],
      'avg_valid_bal_acc':[],'avg_valid_acc':[],  'avg_valid_auc':[], 'avg_valid_tp':[], 'avg_valid_tn':[], 'avg_valid_fp':[], 'avg_valid_fn':[], 'avg_valid_sens':[], 'avg_valid_spec':[], 'avg_valid_ppv':[], 'avg_valid_npv':[], 'avg_valid_f1':[]}

        std_results = {'model':[], 
       'std_train_bal_acc':[], 'std_train_acc':[],  'std_train_auc':[], 'std_train_tp':[], 'std_train_tn':[], 'std_train_fp':[], 'std_train_fn':[],  'std_train_sens':[], 'std_train_spec':[], 'std_train_ppv':[], 'std_train_npv':[], 'std_train_f1':[],
      'std_valid_bal_acc':[],'std_valid_acc':[], 'std_valid_auc':[], 'std_valid_tp':[], 'std_valid_tn':[], 'std_valid_fp':[], 'std_valid_fn':[], 'std_valid_sens':[], 'std_valid_spec':[], 'std_valid_ppv':[], 'std_valid_npv':[], 'std_valid_f1':[]}
        
        for model_name, model in self.models.items():
            #print(f"Running {model_name}")
            run = 1
            # Track run results
            runs_dict = {'run':[], 
                      't_bal_acc':[],'t_acc':[], 't_auc':[], 't_tp':[], 't_tn':[], 't_fp':[], 't_fn':[], 't_sens':[], 't_spec': [], 't_ppv':[], 't_npv':[], 't_f1':[],
                      'v_bal_acc':[],'v_acc':[], 'v_auc':[], 'v_tp':[], 'v_tn':[], 'v_fp':[], 'v_fn':[],'v_sens':[], 'v_spec':[], 'v_ppv':[], 'v_npv':[], 'v_f1': []}
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
                runs_dict['t_acc'].append(run_results[1])
                runs_dict['t_auc'].append(run_results[2])
                runs_dict['t_tp'].append(run_results[3])
                runs_dict['t_tn'].append(run_results[4])
                runs_dict['t_fp'].append(run_results[5])
                runs_dict['t_fn'].append(run_results[6])
                runs_dict['t_sens'].append(run_results[7])
                runs_dict['t_spec'].append(run_results[8])
                runs_dict['t_ppv'].append(run_results[9])
                runs_dict['t_npv'].append(run_results[10])
                runs_dict['t_f1'].append(run_results[11])

                runs_dict['v_bal_acc'].append(run_results[12])
                runs_dict['v_acc'].append(run_results[13])
                runs_dict['v_auc'].append(run_results[14])
                runs_dict['v_tp'].append(run_results[15])
                runs_dict['v_tn'].append(run_results[16])
                runs_dict['v_fp'].append(run_results[17])
                runs_dict['v_fn'].append(run_results[18])
                runs_dict['v_sens'].append(run_results[19])
                runs_dict['v_spec'].append(run_results[20])
                runs_dict['v_ppv'].append(run_results[21])
                runs_dict['v_npv'].append(run_results[22])
                runs_dict['v_f1'].append(run_results[23])

                run +=1

            runs_df = pd.DataFrame(runs_dict)

            # Calculate avg scores across all runs
            exp_results['model'].append(model_name)
            exp_results['avg_train_bal_acc'].append(runs_df['t_bal_acc'].mean())
            exp_results['avg_train_acc'].append(runs_df['t_acc'].mean())
            exp_results['avg_train_auc'].append(runs_df['t_auc'].mean())
            exp_results['avg_train_tp'].append(runs_df['t_tp'].mean())
            exp_results['avg_train_tn'].append(runs_df['t_tn'].mean())
            exp_results['avg_train_fp'].append(runs_df['t_fp'].mean())
            exp_results['avg_train_fn'].append(runs_df['t_fn'].mean())
            exp_results['avg_train_sens'].append(runs_df['t_sens'].mean())
            exp_results['avg_train_spec'].append(runs_df['t_spec'].mean())
            exp_results['avg_train_ppv'].append(runs_df['t_ppv'].mean())
            exp_results['avg_train_npv'].append(runs_df['t_npv'].mean())
            exp_results['avg_train_f1'].append(runs_df['t_f1'].mean())

            exp_results['avg_valid_bal_acc'].append(runs_df['v_bal_acc'].mean())
            exp_results['avg_valid_acc'].append(runs_df['v_acc'].mean())
            exp_results['avg_valid_auc'].append(runs_df['v_auc'].mean())
            exp_results['avg_valid_tp'].append(runs_df['v_tp'].mean())
            exp_results['avg_valid_tn'].append(runs_df['v_tn'].mean())
            exp_results['avg_valid_fp'].append(runs_df['v_fp'].mean())
            exp_results['avg_valid_fn'].append(runs_df['v_fn'].mean())           
            exp_results['avg_valid_sens'].append(runs_df['v_sens'].mean())
            exp_results['avg_valid_spec'].append(runs_df['v_spec'].mean())
            exp_results['avg_valid_ppv'].append(runs_df['v_ppv'].mean())
            exp_results['avg_valid_npv'].append(runs_df['v_npv'].mean())
            exp_results['avg_valid_f1'].append(runs_df['v_f1'].mean())

            std_results['model'].append(model_name)
            std_results['std_train_bal_acc'].append(runs_df['t_bal_acc'].std())
            std_results['std_train_acc'].append(runs_df['t_acc'].std())
            std_results['std_train_auc'].append(runs_df['t_auc'].std())
            std_results['std_train_tp'].append(runs_df['t_tp'].std())
            std_results['std_train_tn'].append(runs_df['t_tn'].std())
            std_results['std_train_fp'].append(runs_df['t_fp'].std())
            std_results['std_train_fn'].append(runs_df['t_fn'].std())
            std_results['std_train_sens'].append(runs_df['t_sens'].std())
            std_results['std_train_spec'].append(runs_df['t_spec'].std())
            std_results['std_train_ppv'].append(runs_df['t_ppv'].std())
            std_results['std_train_npv'].append(runs_df['t_npv'].std())
            std_results['std_train_f1'].append(runs_df['t_f1'].std())

            std_results['std_valid_bal_acc'].append(runs_df['v_bal_acc'].std())
            std_results['std_valid_acc'].append(runs_df['v_acc'].std())
            std_results['std_valid_auc'].append(runs_df['v_auc'].std())
            std_results['std_valid_tp'].append(runs_df['v_tp'].std())
            std_results['std_valid_tn'].append(runs_df['v_tn'].std())
            std_results['std_valid_fp'].append(runs_df['v_fp'].std())
            std_results['std_valid_fn'].append(runs_df['v_fn'].std())
            std_results['std_valid_sens'].append(runs_df['v_sens'].std())
            std_results['std_valid_spec'].append(runs_df['v_spec'].std())
            std_results['std_valid_ppv'].append(runs_df['v_ppv'].std())
            std_results['std_valid_npv'].append(runs_df['v_npv'].std())
            std_results['std_valid_f1'].append(runs_df['v_f1'].std())

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
        f.write(f"Evaluated using {self.X_type} and {self.y_type}")



def main(eval_type: str, eval_name: str):

    startTime = datetime.datetime.now()
  
    assert eval_type in ['full', 'full_enet', 'over', 'over_enet', 'full_qids', 'full_qlesq', 'full_qidsqlesq', 'full_noqidsqlesq', 'full_noqidsqlesq_enet', 'over_qids', 'over_qlesq', 'over_qidsqlesq', 'over_noqidsqlesq', 'over_noqidsqlesq_enet'], "eval_type (1st argument) was not valid. The only 3 options are 'full', 'over', and 'canbind'."

    if eval_type == "full": 
        x_data = "X_train_77"
        models = full_feat_models # these are optimized models discovered in GridSearchCV and are all stored in globals.py

    elif eval_type == "full_enet":
        x_data = "X_train_77_enet"

        models = full_enet_feat_models # haven't done Grid search on this yet so just use default full models

    elif eval_type == "over":
        x_data = "X_train_77_over"

        models = overlapping_feat_models
    
    elif eval_type == "over_enet":
        x_data = "X_train_77_over_enet"
        models = overlapping_enet_feat_models
    
    elif eval_type == "full_qids":
        x_data = "X_train_77_qids"
        models = qids_models
    
    elif eval_type == "full_qlesq":
        x_data = "X_train_77_qlesq"
        models = qlesq_models

    elif eval_type == "full_qidsqlesq":
        x_data = "X_train_77_qidsqlesq"
        models = qidsqlesq_models
    
    elif eval_type == "full_noqidsqlesq":
        x_data = "X_train_77_noqidsqlesq"
        models = noqidsqlesq_models

    elif eval_type == "full_noqidsqlesq_enet":
        x_data = "X_train_77_noqidsqlesq_enet"
        models = noqidsqlesq_enet_models

    elif eval_type == "over_qids":
        x_data = "X_train_77_over_qids"
        models = over_qids_models
    
    elif eval_type == "over_qlesq":
        x_data = "X_train_77_over_qlesq"
        models = over_qlesq_models

    elif eval_type == "over_qidsqlesq":
        x_data = "X_train_77_over_qidsqlesq"
        models = over_qidsqlesq_models
    
    elif eval_type == "over_noqidsqlesq":
        x_data = "X_train_77_over_noqidsqlesq"
        models = over_noqidsqlesq_models

    elif eval_type == "over_noqidsqlesq_enet":
        x_data = "X_train_77_over_noqidsqlesq_enet"

    y_data = "y_train_77"

    

    x_path = os.path.join(DATA_MODELLING_FOLDER, x_data)
    y_path = os.path.join(DATA_MODELLING_FOLDER, y_data)

    print(x_path)
    print(y_path)
   
    X = pd.read_csv(x_path + ".csv")
    y = pd.read_csv(y_path + ".csv")


    y.columns = ['subjectkey', 'target']

    exp_1 = experiment_manager(X, y, x_data, y_data)

    exp_1.models = models

    # full Feature models
    # exp_1.models = {'Dummy Classification': ('dummy', DummyClassifier(strategy = 'stratified')), 
    #       'Random Forest' : ('rf', RandomForestClassifier(class_weight = 'balanced', max_depth = 2, max_features = 'sqrt')),
    #       'Logistic Regression': ('lr', LogisticRegression(solver = 'saga', class_weight = 'balanced', penalty = 'elasticnet', max_iter = 1000,  C = 0.096, tol = 0.1, l1_ratio = 0.8)),
    #       'KNearest Neighbors': ('knn', KNeighborsClassifier(n_neighbors = 15, p = 1, weights = 'uniform')),
    #       'Support Vector Machine':('svc', SVC(class_weight = 'balanced', C = 1, gamma = 'auto', probability = True) )}

    # Overlapping models
    # exp_1.models = {'Dummy Classification': ('dummy', DummyClassifier(strategy = 'stratified')),
    #                 'Logistic Regression': ('lr', LogisticRegression(solver = 'saga', class_weight = 'balanced', penalty = 'elasticnet', max_iter = 1000,  C = 0.091, tol = 0.1, l1_ratio = 0.9)),
    #                 'Random Forest' :('rf', RandomForestClassifier(class_weight = 'balanced', max_depth = 2, max_features = 'sqrt')),
    #                 'KNearest Neighbors' :('knn', KNeighborsClassifier(n_neighbors = 1, p = 1, weights = 'uniform')),
    #                 'SVC' :('svc', SVC(class_weight = 'balanced', C= 1, gamma= 'scale', probability = True))}

    exp_1.run_experiment()

    print(exp_1.avg_results)
    print(exp_1.std_results)

    exp_1.store_results(EXPERIMENT_RESULTS, eval_name)

    

if __name__ == "__main__":
    typer.run(main)