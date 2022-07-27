# run_experiment

# Basics
import pandas as pd
import numpy as np
import datetime
import pickle
import typer
import os

# Import paths
from globals import DATA_MODELLING_FOLDER, EVALUATION_RESULTS, full_feat_models, overlapping_feat_models, full_feat_models_rfe

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
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

class evaluation_manager():
    
    def __init__(self, X_train, y_train, X_test, y_test, x_train_data, y_train_data, x_test_data, y_test_data, eval_name, EVALUATION_RESULTS):
        
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
        self.results_path = EVALUATION_RESULTS
        self.eval_name = eval_name

        self.out_path = os.path.join(self.results_path, self.eval_name)
        self.model_path = self.out_path + "/models/"

        if os.path.isdir(self.out_path):
            raise Exception("Name already exists")
        else:
            os.mkdir(self.out_path + "/")

        
        os.mkdir(self.out_path + "/models/")
        

        
    def eval_run(self, pipe_model, model_name, run):    
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

        # Obtain training sensitivity/recall, specificity, PPV/precision, and NPV
        train_tn, train_fp, train_fn, train_tp = confusion_matrix(self.y_train, training_predictions).ravel()
        train_spec = train_tn/(train_tn + train_fp)
        train_sens = train_tp/(train_tp + train_fn)
        train_prec = train_tp/(train_tp + train_fp)
        train_npv = train_tn/(train_tn + train_fn)
        train_f1 = f1_score(self.y_train, training_predictions)
        

        # Obtain validation sensitivity, specificity, PPV/precision, and NPV
        test_tn, test_fp, test_fn, test_tp = confusion_matrix(self.y_test, test_predictions).ravel()
        test_spec = test_tn/(test_tn + test_fp)
        test_sens = test_tp/(test_tp + test_fn)
        test_prec = test_tp/(test_tp + test_fp)
        test_npv = test_tn/(test_tn + test_fn)
        test_f1 = f1_score(self.y_test, test_predictions)

        # Store scores for the current split
        train_bal_acc = balanced_accuracy_score(self.y_train, training_predictions)
        test_bal_acc = balanced_accuracy_score(self.y_test, test_predictions)

        train_acc = accuracy_score(self.y_train, training_predictions)
        test_acc = accuracy_score(self.y_test, test_predictions)

        # Feature Importances 
        print(pipe_model.steps[1][1])
        if model_name == 'Dummy_Classification' or model_name == "KNearest_Neighbors" or model_name == "Support_Vector_Machine":
            FI = np.zeros(len(self.X_train.columns))
        elif model_name == 'Logistic_Regression':
            FI = pipe_model.steps[1][1].coef_.flatten()
        elif model_name == 'Random_Forest' or model_name == 'Gradient Boosting Classifier':
            FI = pipe_model.steps[1][1].feature_importances_.flatten()
        else: raise Exception("model_name doesn't match options for FI")

        # Store model
        pickle.dump(pipe_model, open(self.model_path + "/" + model_name + "_" + str(run), 'wb'))

        return train_bal_acc, train_acc, train_auc, train_tp, train_tn, train_fp, train_fn, train_sens, train_spec, train_prec, train_npv, train_f1, test_bal_acc, test_acc, test_auc, test_tp, test_tn, test_fp, test_fn, test_sens, test_spec, test_prec, test_npv, test_f1, FI

    def process_feature_importances(self, FI_array, model_name):
 
        FI_array = FI_array/self.runs
        feat_importance_dict = {"Feature": self.X_train.columns, "Weight": FI_array}
        self.FI_df = pd.DataFrame(feat_importance_dict).sort_values(by = "Weight", ascending = False)
        self.FI_df.to_csv(self.out_path + "/" + model_name + "_FI.csv")
        return
    
    def run_evaluation(self):
        startTime = datetime.datetime.now()
        
        assert len(self.models) >= 1, "No models to run test for!"
        exp_results = {'model':[], 
       'avg_train_bal_acc':[], 'avg_train_acc':[], 'avg_train_auc':[],'avg_train_tp':[], 'avg_train_tn':[], 'avg_train_fp':[], 'avg_train_fn':[], 'avg_train_sens':[], 'avg_train_spec':[], 'avg_train_ppv':[],'avg_train_npv':[], 'avg_train_f1':[],
      'avg_test_bal_acc':[], 'avg_test_acc':[], 'avg_test_auc':[], 'avg_test_tp':[], 'avg_test_tn':[], 'avg_test_fp':[], 'avg_test_fn':[], 'avg_test_sens':[], 'avg_test_spec':[], 'avg_test_ppv':[], 'avg_test_npv':[], 'avg_test_f1':[]}

        std_results = {'model':[], 
       'std_train_bal_acc':[],'std_train_acc':[], 'std_train_auc':[],'std_train_tp':[], 'std_train_tn':[], 'std_train_fp':[], 'std_train_fn':[], 'std_train_sens':[], 'std_train_spec':[], 'std_train_ppv':[], 'std_train_npv':[], 'std_train_f1':[],
      'std_test_bal_acc':[], 'std_test_acc':[], 'std_test_auc':[], 'std_test_tp':[], 'std_test_tn':[], 'std_test_fp':[], 'std_test_fn':[], 'std_test_sens':[], 'std_test_spec':[], 'std_test_ppv':[], 'std_test_npv':[], 'std_test_f1':[]}
        
        for model_name, model in self.models.items():
            #print(f"Running {model_name}")
            run = 1
            # Track run results
            runs_dict = {'run':[], 
                      'train_bal_acc':[],'train_acc':[], 'train_auc':[],'train_tp':[], 'train_tn':[], 'train_fp':[], 'train_fn':[], 'train_sens':[], 'train_spec': [], 'train_ppv':[], 'train_npv':[], 'train_f1':[],
                      'test_bal_acc':[],'test_acc':[], 'test_auc':[], 'test_tp':[], 'test_tn':[], 'test_fp':[], 'test_fn':[], 'test_sens':[],'test_sens':[], 'test_spec':[], 'test_ppv':[], 'test_npv':[], 'test_f1':[]}

            # Will track feat importances across runs
            feat_importances = np.zeros(len(self.X_train.columns))

            # Run experiment multiple times (ex. 100)
            for r in range(self.runs):

                # Create a pipline with scaling procedure and model of interest
                pipe = Pipeline([('scaler', MinMaxScaler()), model])

                runs_dict['run'].append(run)

                run_results = self.eval_run(pipe, model_name, run)
                
                # Unloading tuple in order of: 
                # avg_t_bal_acc, avg_t_auc ,avg_t_sens, avg_t_spec, avg_t_prec, avg_t_npv, (0-7)
                # avg_v_bal_acc, avg_v_auc, avg_v_sens, avg_v_spec, avg_v_prec, avg_v_npv (8-15)
                # feature importance (16)
                
                runs_dict['train_bal_acc'].append(run_results[0])
                runs_dict['train_acc'].append(run_results[1])
                runs_dict['train_auc'].append(run_results[2])
                runs_dict['train_tp'].append(run_results[3])
                runs_dict['train_tn'].append(run_results[4])
                runs_dict['train_fp'].append(run_results[5])
                runs_dict['train_fn'].append(run_results[6])
                runs_dict['train_sens'].append(run_results[7])
                runs_dict['train_spec'].append(run_results[8])
                runs_dict['train_ppv'].append(run_results[9])
                runs_dict['train_npv'].append(run_results[10])
                runs_dict['train_f1'].append(run_results[11])

                runs_dict['test_bal_acc'].append(run_results[12])
                runs_dict['test_acc'].append(run_results[13])
                runs_dict['test_auc'].append(run_results[14])
                runs_dict['test_tp'].append(run_results[15])
                runs_dict['test_tn'].append(run_results[16])
                runs_dict['test_fp'].append(run_results[17])
                runs_dict['test_fn'].append(run_results[18])
                runs_dict['test_sens'].append(run_results[19])
                runs_dict['test_spec'].append(run_results[20])
                runs_dict['test_ppv'].append(run_results[21])
                runs_dict['test_npv'].append(run_results[22])
                runs_dict['test_f1'].append(run_results[23])

                feat_importances += run_results[24]

                run +=1

            runs_df = pd.DataFrame(runs_dict)

            # Calculate avg scores across all runs
            exp_results['model'].append(model_name)
            exp_results['avg_train_bal_acc'].append(runs_df['train_bal_acc'].mean())
            exp_results['avg_train_acc'].append(runs_df['train_acc'].mean())
            exp_results['avg_train_auc'].append(runs_df['train_auc'].mean())
            exp_results['avg_train_tp'].append(runs_df['train_tp'].mean())
            exp_results['avg_train_tn'].append(runs_df['train_tn'].mean())
            exp_results['avg_train_fp'].append(runs_df['train_fp'].mean())
            exp_results['avg_train_fn'].append(runs_df['train_fn'].mean())
            exp_results['avg_train_sens'].append(runs_df['train_sens'].mean())
            exp_results['avg_train_spec'].append(runs_df['train_spec'].mean())
            exp_results['avg_train_ppv'].append(runs_df['train_ppv'].mean())
            exp_results['avg_train_npv'].append(runs_df['train_npv'].mean())
            exp_results['avg_train_f1'].append(runs_df['train_f1'].mean())

            exp_results['avg_test_bal_acc'].append(runs_df['test_bal_acc'].mean())
            exp_results['avg_test_acc'].append(runs_df['test_acc'].mean())
            exp_results['avg_test_auc'].append(runs_df['test_auc'].mean())
            exp_results['avg_test_tp'].append(runs_df['test_tp'].mean())
            exp_results['avg_test_tn'].append(runs_df['test_tn'].mean())
            exp_results['avg_test_fp'].append(runs_df['test_fp'].mean())
            exp_results['avg_test_fn'].append(runs_df['test_fn'].mean())
            exp_results['avg_test_sens'].append(runs_df['test_sens'].mean())
            exp_results['avg_test_spec'].append(runs_df['test_spec'].mean())
            exp_results['avg_test_ppv'].append(runs_df['test_ppv'].mean())
            exp_results['avg_test_npv'].append(runs_df['test_npv'].mean())
            exp_results['avg_test_f1'].append(runs_df['test_f1'].mean())

            # Calculate avg standard deviations across all runs
            std_results['model'].append(model_name)
            std_results['std_train_bal_acc'].append(runs_df['train_bal_acc'].std())
            std_results['std_train_acc'].append(runs_df['train_acc'].std())
            std_results['std_train_auc'].append(runs_df['train_auc'].std())
            std_results['std_train_tp'].append(runs_df['train_tp'].std())
            std_results['std_train_tn'].append(runs_df['train_tn'].std())
            std_results['std_train_fp'].append(runs_df['train_fp'].std())
            std_results['std_train_fn'].append(runs_df['train_fn'].std())
            std_results['std_train_sens'].append(runs_df['train_sens'].std())
            std_results['std_train_spec'].append(runs_df['train_spec'].std())
            std_results['std_train_ppv'].append(runs_df['train_ppv'].std())
            std_results['std_train_npv'].append(runs_df['train_npv'].std())
            std_results['std_train_f1'].append(runs_df['train_f1'].std())

            std_results['std_test_bal_acc'].append(runs_df['test_bal_acc'].std())
            std_results['std_test_acc'].append(runs_df['test_acc'].std())
            std_results['std_test_auc'].append(runs_df['test_auc'].std())
            std_results['std_test_tp'].append(runs_df['test_tp'].std())
            std_results['std_test_tn'].append(runs_df['test_tn'].std())
            std_results['std_test_fp'].append(runs_df['test_fp'].std())
            std_results['std_test_fn'].append(runs_df['test_fn'].std())
            std_results['std_test_sens'].append(runs_df['test_sens'].std())
            std_results['std_test_spec'].append(runs_df['test_spec'].std())
            std_results['std_test_ppv'].append(runs_df['test_ppv'].std())
            std_results['std_test_npv'].append(runs_df['test_npv'].std())
            std_results['std_test_f1'].append(runs_df['test_f1'].std())

            self.process_feature_importances(feat_importances, model_name)

        self.avg_results = pd.DataFrame(exp_results)
        self.std_results = pd.DataFrame(std_results)
        

        print(f"Completed in: {datetime.datetime.now() - startTime}")

    def store_results(self):
        

        merged_df = self.avg_results.merge(self.std_results, on = 'model')

        self.avg_results.to_csv(self.out_path + "/" + self.eval_name + "_avg.csv" ,index = False)
        self.std_results.to_csv(self.out_path + "/" + self.eval_name + "_std.csv" ,index = False)
        merged_df.to_csv(self.out_path + "/" + self.eval_name + "_merged.csv" ,index = False)

        f = open(os.path.join(self.out_path, self.eval_name + '.txt'), 'w')
        f.write(f"Model used: {self.models}\n\n")
        f.write(f"Trained on  {self.X_train_type} and {self.y_train_type}\n\n")
        f.write(f"Evaluated on  {self.X_test_type} and {self.y_test_type}")


def main(eval_type : str, eval_name : str):

    startTime = datetime.datetime.now()

    assert eval_type in ['full', 'full_enet', 'full_rfe', 'over', 'canbind'], "eval_type (1st argument) was not valid. The only 3 options are 'full', 'over', and 'canbind'."

    if eval_type == "full": 
        x_train_data = "X_train_77"
        y_train_data = "y_train_77"
        x_test_data = "X_test_77"
        y_test_data = "y_test_77"

        models = full_feat_models # these are optimized models discovered in GridSearchCV and are all stored in globals.py

    elif eval_type == "full_enet":
        x_train_data = "X_train_77_enet"
        y_train_data = "y_train_77"
        x_test_data = "X_test_77_enet"
        y_test_data = "y_test_77"

        models = full_feat_models # haven't done Grid search on this yet so just use default full models

    elif eval_type == "full_rfe":
        x_train_data = "X_train_77_rfe"
        y_train_data = "y_train_77"
        x_test_data = "X_test_77_rfe"
        y_test_data = "y_test_77"

        models = full_feat_models_rfe

    elif eval_type == "over":
        x_train_data = "X_train_77_over"
        y_train_data = "y_train_77"
        x_test_data = "X_test_77_over"
        y_test_data = "y_test_77"

        models = overlapping_feat_models
    
    elif eval_type == "canbind":
        x_train_data = "X_77_qlesq_sr__final_extval"
        y_train_data = "y_qlesq_77__final__targets"

        x_test_data = "X_test_cb_extval"
        y_test_data = "canbind_qlesq_y__targets"

        models = overlapping_feat_models

    x_train_path = os.path.join(DATA_MODELLING_FOLDER, x_train_data)
    y_train_path = os.path.join(DATA_MODELLING_FOLDER, y_train_data)
    x_test_path = os.path.join(DATA_MODELLING_FOLDER, x_test_data)
    y_test_path = os.path.join(DATA_MODELLING_FOLDER, y_test_data)
   
    X_train = pd.read_csv(x_train_path + ".csv").set_index('subjectkey')
    y_train = pd.read_csv(y_train_path + ".csv").set_index('subjectkey')

    X_test = pd.read_csv(x_test_path + ".csv").set_index('subjectkey')
    y_test = pd.read_csv(y_test_path + ".csv").set_index('subjectkey')

    # Some processing was left out until the end for the canbind dataset
    if eval_type == "canbind":
        
        # Drops 5 rows that weren't shared by both dfs. The discrepancy is due to selection criteria in canbind_ygen.py applied on the y df.
        X_test = X_test[X_test.index.isin(list(X_test.index.difference(y_test.index))) == False]
        y_test = y_test[y_test.index.isin(list(y_test.index.difference(X_test.index))) == False]

        # For feature selection, adjust canbind dataset to have only selected columns
        if X_train.shape[1] != X_test.shape[1]:
            selected_cols = X_train.columns
            X_test = X_test[selected_cols]

        y_train = y_train[['qlesq_QoL_threshold']]
        y_test = y_test[['qlesq_QoL_threshold']]



    y_train.columns = ['target']
    y_test.columns = ['target']

    eval_1 = evaluation_manager(X_train, y_train, X_test, y_test, x_train_data, y_train_data, x_test_data, y_test_data, eval_name, EVALUATION_RESULTS)

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

    eval_1.store_results()


if __name__ == "__main__":
    typer.run(main)