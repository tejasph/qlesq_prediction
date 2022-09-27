# globals
# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os

# DATA_MODELLING_FOLDER = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\data\modelling"
# GRID_SEARCH_RESULTS = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\grid_search"
# EXPERIMENT_RESULTS = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\experiments"
# EVALUATION_RESULTS = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations"
# ENET_RESULTS = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\enet"
# STAT_RESULTS = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\stats"

PROJECT_DIR = r"C:\Users\jjnun\PycharmProjects\qlesq_prediction"
DATA_MODELLING_FOLDER = os.path.join(PROJECT_DIR, r"data\modelling")
GRID_SEARCH_RESULTS = os.path.join(PROJECT_DIR, r"results\grid_search")
EXPERIMENT_RESULTS = os.path.join(PROJECT_DIR, r"results\experiments")
EVALUATION_RESULTS = os.path.join(PROJECT_DIR, r"results\evaluations")
ENET_RESULTS = os.path.join(PROJECT_DIR, r"results\enet")
STAT_RESULTS = os.path.join(PROJECT_DIR, r"results\stats")

# Excellent article on class imbalance management in GBDT (https://towardsdatascience.com/practical-tips-for-class-imbalance-in-binary-classification-6ee29bcdb8a7)


### New model parameters (Aug 2022)

# Full
full_feat_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy='stratified')),
                    'Random_Forest': ('rf',
                                      RandomForestClassifier(class_weight='balanced', criterion='entropy', max_depth=2,
                                                             max_features='sqrt', n_estimators=500)),
                    'Logistic_Regression': ('lr', LogisticRegression(solver='liblinear', class_weight='balanced',
                                                                     penalty='l2', C=0.01, tol=0.01)),
                    'Elastic_Net': ('en',
                                    LogisticRegression(solver='saga', class_weight='balanced', penalty='elasticnet',
                                                       C=0.1, l1_ratio=1.0, tol=0.001)),
                    'KNearest_Neighbors': ('knn', KNeighborsClassifier(n_neighbors=17, p=1, weights='uniform')),
                    'Support_Vector_Machine': (
                    'svc', SVC(class_weight='balanced', C=1, gamma='auto', probability=True)),
                    'Gradient Boosting Classifier': ('gbdt', GradientBoostingClassifier(learning_rate=0.1, max_depth=3,
                                                                                        max_features='sqrt',
                                                                                        n_estimators=500,
                                                                                        subsample=0.9))}

full_enet_feat_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy='stratified')),
                         'Random_Forest': ('rf', RandomForestClassifier(class_weight='balanced', criterion='entropy',
                                                                        max_depth=4, max_features='log2',
                                                                        n_estimators=500)),
                         'Logistic_Regression': ('lr', LogisticRegression(solver='liblinear', class_weight='balanced',
                                                                          penalty='l2', C=100, tol=0.1)),
                         'KNearest_Neighbors': ('knn', KNeighborsClassifier(n_neighbors=3, p=1, weights='uniform')),
                         'Support_Vector_Machine': (
                         'svc', SVC(class_weight='balanced', C=1, gamma='scale', probability=True)),
                         'Gradient Boosting Classifier': ('gbdt',
                                                          GradientBoostingClassifier(learning_rate=1, max_depth=2,
                                                                                     max_features="sqrt",
                                                                                     n_estimators=500, subsample=0.9))}

# Overlapping
overlapping_feat_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy='stratified')),
                           'Random_Forest': ('rf', RandomForestClassifier(class_weight='balanced', criterion='gini',
                                                                          max_depth=3, max_features=0.1,
                                                                          n_estimators=500)),
                           'Logistic_Regression': ('lr', LogisticRegression(solver='liblinear', class_weight='balanced',
                                                                            penalty='l2', C=0.1, tol=0.1)),
                           'Elastic_Net': ('en', LogisticRegression(solver='saga', class_weight='balanced',
                                                                    penalty='elasticnet', C=0.1, l1_ratio=0.0,
                                                                    tol=0.01)),
                           'KNearest_Neighbors': ('knn', KNeighborsClassifier(n_neighbors=1, p=1, weights='uniform')),
                           'Support_Vector_Machine': (
                           'svc', SVC(class_weight='balanced', C=10, gamma='auto', probability=True)),
                           'Gradient Boosting Classifier': ('gbdt',
                                                            GradientBoostingClassifier(learning_rate=0.1, max_depth=2,
                                                                                       max_features=0.1,
                                                                                       n_estimators=500,
                                                                                       subsample=1.0))}

overlapping_enet_feat_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy='stratified')),
                                'Random_Forest': ('rf',
                                                  RandomForestClassifier(class_weight='balanced', criterion='gini',
                                                                         max_depth=4, max_features='log2',
                                                                         n_estimators=500)),
                                'Logistic_Regression': ('lr',
                                                        LogisticRegression(solver='liblinear', class_weight='balanced',
                                                                           penalty='l2', C=0.1, tol=0.1)),
                                'KNearest_Neighbors': (
                                'knn', KNeighborsClassifier(n_neighbors=7, p=1, weights='distance')),
                                'Support_Vector_Machine': (
                                'svc', SVC(class_weight='balanced', C=10, gamma='auto', probability=True)),
                                'Gradient Boosting Classifier': ('gbdt', GradientBoostingClassifier(learning_rate=1,
                                                                                                    max_depth=2,
                                                                                                    max_features=0.2,
                                                                                                    n_estimators=500,
                                                                                                    subsample=1.0))}

# QLESQ vs QIDS vs both vs none

qlesq_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy='stratified')),
                'Random_Forest': ('rf',
                                  RandomForestClassifier(class_weight='balanced', criterion='entropy', max_depth=3,
                                                         max_features='sqrt', n_estimators=500)),
                'Logistic_Regression': (
                'lr', LogisticRegression(solver='liblinear', class_weight='balanced', penalty='l2', C=0.1, tol=0.1)),
                'Elastic_Net': ('en',
                                LogisticRegression(solver='saga', class_weight='balanced', penalty='elasticnet', C=0.1,
                                                   l1_ratio=0, tol=0.1)),
                'KNearest_Neighbors': ('knn', KNeighborsClassifier(n_neighbors=1, p=1, weights='uniform')),
                'Support_Vector_Machine': ('svc', SVC(class_weight='balanced', C=1, gamma='auto', probability=True)),
                'Gradient Boosting Classifier': ('gbdt', GradientBoostingClassifier(learning_rate=1, max_depth=4,
                                                                                    max_features=0.2, n_estimators=500,
                                                                                    subsample=0.9))}

qids_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy='stratified')),
               'Random_Forest': ('rf', RandomForestClassifier(class_weight='balanced', criterion='entropy', max_depth=2,
                                                              max_features=0.2, n_estimators=500)),
               'Logistic_Regression': (
               'lr', LogisticRegression(solver='liblinear', class_weight='balanced', penalty='l2', C=0.1, tol=0.1)),
               'Elastic_Net': ('en',
                               LogisticRegression(solver='saga', class_weight='balanced', penalty='elasticnet', C=0.01,
                                                  l1_ratio=0.1, tol=0.01)),
               'KNearest_Neighbors': ('knn', KNeighborsClassifier(n_neighbors=7, p=1, weights='distance')),
               'Support_Vector_Machine': ('svc', SVC(class_weight='balanced', C=0.1, gamma='scale', probability=True)),
               'Gradient Boosting Classifier': ('gbdt', GradientBoostingClassifier(learning_rate=0.1, max_depth=2,
                                                                                   max_features='sqrt',
                                                                                   n_estimators=500, subsample=0.9))}

qidsqlesq_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy='stratified')),
                    'Random_Forest': ('rf',
                                      RandomForestClassifier(class_weight='balanced', criterion='gini', max_depth=4,
                                                             max_features=0.33, n_estimators=500)),
                    'Logistic_Regression': ('lr', LogisticRegression(solver='liblinear', class_weight='balanced',
                                                                     penalty='l2', C=0.1, tol=0.1)),
                    'Elastic_Net': ('en',
                                    LogisticRegression(solver='saga', class_weight='balanced', penalty='elasticnet',
                                                       C=0.1, l1_ratio=0, tol=0.01)),
                    'KNearest_Neighbors': ('knn', KNeighborsClassifier(n_neighbors=1, p=2, weights='uniform')),
                    'Support_Vector_Machine': (
                    'svc', SVC(class_weight='balanced', C=1, gamma='auto', probability=True)),
                    'Gradient Boosting Classifier': ('gbdt', GradientBoostingClassifier(learning_rate=0.1, max_depth=2,
                                                                                        max_features=0.33,
                                                                                        n_estimators=500,
                                                                                        subsample=0.7))}

noqidsqlesq_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy='stratified')),
                      'Random_Forest': ('rf', RandomForestClassifier(class_weight='balanced', criterion='entropy',
                                                                     max_depth=2, max_features=0.33, n_estimators=500)),
                      'Logistic_Regression': ('lr', LogisticRegression(solver='liblinear', class_weight='balanced',
                                                                       penalty='l2', C=10, tol=0.01)),
                      'Elastic_Net': ('en',
                                      LogisticRegression(solver='saga', class_weight='balanced', penalty='elasticnet',
                                                         C=0.01, l1_ratio=0.2, tol=0.1)),
                      'KNearest_Neighbors': ('knn', KNeighborsClassifier(n_neighbors=7, p=1, weights='uniform')),
                      'Support_Vector_Machine': (
                      'svc', SVC(class_weight='balanced', C=1, gamma='auto', probability=True)),
                      'Gradient Boosting Classifier': ('gbdt', GradientBoostingClassifier(learning_rate=1, max_depth=2,
                                                                                          max_features=0.33,
                                                                                          n_estimators=500,
                                                                                          subsample=1.0))}

noqidsqlesq_enet_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy='stratified')),
                           'Random_Forest': ('rf', RandomForestClassifier(class_weight='balanced', criterion='entropy',
                                                                          max_depth=3, max_features='log2',
                                                                          n_estimators=500)),
                           'Logistic_Regression': ('lr', LogisticRegression(solver='liblinear', class_weight='balanced',
                                                                            penalty='l2', C=1, tol=0.1)),
                           'KNearest_Neighbors': ('knn', KNeighborsClassifier(n_neighbors=1, p=1, weights='uniform')),
                           'Support_Vector_Machine': (
                           'svc', SVC(class_weight='balanced', C=10, gamma='auto', probability=True)),
                           'Gradient Boosting Classifier': ('gbdt',
                                                            GradientBoostingClassifier(learning_rate=0.1, max_depth=2,
                                                                                       max_features=0.2,
                                                                                       n_estimators=500,
                                                                                       subsample=0.7))}

over_qids_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy='stratified')),
                    'Random_Forest': ('rf',
                                      RandomForestClassifier(class_weight='balanced', criterion='entropy', max_depth=3,
                                                             max_features='sqrt', n_estimators=500)),
                    'Logistic_Regression': (
                    'lr', LogisticRegression(solver='liblinear', class_weight='balanced', penalty='l2', C=1, tol=0.1)),
                    'Elastic_Net': ('en',
                                    LogisticRegression(solver='saga', class_weight='balanced', penalty='elasticnet',
                                                       C=1000, l1_ratio=1.0, tol=0.1)),
                    'KNearest_Neighbors': ('knn', KNeighborsClassifier(n_neighbors=1, p=2, weights='uniform')),
                    'Support_Vector_Machine': (
                    'svc', SVC(class_weight='balanced', C=1, gamma='auto', probability=True)),
                    'Gradient Boosting Classifier': ('gbdt', GradientBoostingClassifier(learning_rate=1, max_depth=3,
                                                                                        max_features=0.2,
                                                                                        n_estimators=500,
                                                                                        subsample=0.7))}

over_qlesq_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy='stratified')),
                     'Random_Forest': ('rf',
                                       RandomForestClassifier(class_weight='balanced', criterion='gini', max_depth=2,
                                                              max_features=0.33, n_estimators=500)),
                     'Logistic_Regression': ('lr', LogisticRegression(solver='liblinear', class_weight='balanced',
                                                                      penalty='l2', C=0.1, tol=0.1)),
                     'Elastic_Net': ('en',
                                     LogisticRegression(solver='saga', class_weight='balanced', penalty='elasticnet',
                                                        C=0.1, l1_ratio=0.1, tol=0.01)),
                     'KNearest_Neighbors': ('knn', KNeighborsClassifier(n_neighbors=3, p=2, weights='uniform')),
                     'Support_Vector_Machine': (
                     'svc', SVC(class_weight='balanced', C=1, gamma='auto', probability=True)),
                     'Gradient Boosting Classifier': ('gbdt', GradientBoostingClassifier(learning_rate=1, max_depth=2,
                                                                                         max_features=0.33,
                                                                                         n_estimators=500,
                                                                                         subsample=1.0))}

over_qidsqlesq_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy='stratified')),
                         'Random_Forest': ('rf', RandomForestClassifier(class_weight='balanced', criterion='gini',
                                                                        max_depth=3, max_features='log2',
                                                                        n_estimators=500)),
                         'Logistic_Regression': ('lr', LogisticRegression(solver='liblinear', class_weight='balanced',
                                                                          penalty='l2', C=0.1, tol=0.1)),
                         'Elastic_Net': ('en', LogisticRegression(solver='saga', class_weight='balanced',
                                                                  penalty='elasticnet', C=0.1, l1_ratio=0.5,
                                                                  tol=0.001)),
                         'KNearest_Neighbors': ('knn', KNeighborsClassifier(n_neighbors=3, p=2, weights='uniform')),
                         'Support_Vector_Machine': (
                         'svc', SVC(class_weight='balanced', C=1, gamma='auto', probability=True)),
                         'Gradient Boosting Classifier': ('gbdt',
                                                          GradientBoostingClassifier(learning_rate=1, max_depth=2,
                                                                                     max_features='log2',
                                                                                     n_estimators=500, subsample=1.0))}

over_noqidsqlesq_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy='stratified')),
                           'Random_Forest': ('rf', RandomForestClassifier(class_weight='balanced', criterion='gini',
                                                                          max_depth=3, max_features=0.1,
                                                                          n_estimators=500)),
                           'Logistic_Regression': ('lr', LogisticRegression(solver='liblinear', class_weight='balanced',
                                                                            penalty='l2', C=1, tol=0.001)),
                           'Elastic_Net': ('en', LogisticRegression(solver='saga', class_weight='balanced',
                                                                    penalty='elasticnet', C=0.1, l1_ratio=0.1,
                                                                    tol=0.1)),
                           'KNearest_Neighbors': ('knn', KNeighborsClassifier(n_neighbors=3, p=1, weights='uniform')),
                           'Support_Vector_Machine': (
                           'svc', SVC(class_weight='balanced', C=1000, gamma='auto', probability=True)),
                           'Gradient Boosting Classifier': ('gbdt',
                                                            GradientBoostingClassifier(learning_rate=10, max_depth=3,
                                                                                       max_features=0.1,
                                                                                       n_estimators=500,
                                                                                       subsample=0.7))}

###Old Variables
# Full  Variables
# full_feat_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy = 'stratified')), 
#       'Random_Forest' : ('rf', RandomForestClassifier(class_weight = 'balanced', max_depth = 2, max_features = 'sqrt')),
#       'Logistic_Regression': ('lr', LogisticRegression(solver = 'saga', class_weight = 'balanced', penalty = 'elasticnet', max_iter = 1000,  C = 0.096, tol = 0.1, l1_ratio = 0.8)),
#       'KNearest_Neighbors': ('knn', KNeighborsClassifier(n_neighbors = 15, p = 1, weights = 'uniform')),
#       'Support_Vector_Machine':('svc', SVC(class_weight = 'balanced', C = 1, gamma = 'auto', probability = True)),
#       'Gradient Boosting Classifier': ('gbdt', GradientBoostingClassifier(max_features = 'sqrt'))}


# Overlapping  Variables

# overlapping_feat_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy = 'stratified')),
#                 'Logistic_Regression': ('lr', LogisticRegression(solver = 'saga', class_weight = 'balanced', penalty = 'elasticnet', max_iter = 1000,  C = 0.091, tol = 0.1, l1_ratio = 0.9)),
#                 'Random_Forest' :('rf', RandomForestClassifier(class_weight = 'balanced', max_depth = 2, max_features = 'sqrt')),
#                 'KNearest_Neighbors' :('knn', KNeighborsClassifier(n_neighbors = 1, p = 1, weights = 'uniform')),
#                 'Support_Vector_Machine' :('svc', SVC(class_weight = 'balanced', C= 1, gamma= 'scale', probability = True))}


# Full Features with RFE (30 features selected)
# full_feat_models_rfe = {'Dummy_Classification': ('dummy', DummyClassifier(strategy = 'stratified')),
#                 'Logistic_Regression': ('lr', LogisticRegression(solver = 'saga', class_weight = 'balanced', penalty = 'elasticnet', max_iter = 1000,  C = 0.105, tol = 0.1, l1_ratio = 0.7)),
#                 'Random_Forest' :('rf', RandomForestClassifier(class_weight = 'balanced', criterion = 'entropy', max_depth = 2, max_features = 0.1, min_impurity_decrease = 0.0, min_samples_leaf = 5, min_samples_split = 2)),
#                 'KNearest_Neighbors' :('knn', KNeighborsClassifier(n_neighbors = 7, p = 1, weights = 'uniform')),
#                 'Support_Vector_Machine' :('svc', SVC(class_weight = 'balanced', C= 1, gamma= 'scale', probability = True))}
######################################################
