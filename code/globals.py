#globals
# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

DATA_MODELLING_FOLDER = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\data\modelling"
GRID_SEARCH_RESULTS = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\grid_search"
EXPERIMENT_RESULTS = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\experiments"
EVALUATION_RESULTS = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations"


# Full  Variables
full_feat_models = {'Dummy Classification': ('dummy', DummyClassifier(strategy = 'stratified')), 
      'Random Forest' : ('rf', RandomForestClassifier(class_weight = 'balanced', max_depth = 2, max_features = 'sqrt')),
      'Logistic Regression': ('lr', LogisticRegression(solver = 'saga', class_weight = 'balanced', penalty = 'elasticnet', max_iter = 1000,  C = 0.096, tol = 0.1, l1_ratio = 0.8)),
      'KNearest Neighbors': ('knn', KNeighborsClassifier(n_neighbors = 15, p = 1, weights = 'uniform')),
      'Support Vector Machine':('svc', SVC(class_weight = 'balanced', C = 1, gamma = 'auto', probability = True))}


# Overlapping  Variables

overlapping_feat_models = {'Dummy Classification': ('dummy', DummyClassifier(strategy = 'stratified')),
                'Logistic Regression': ('lr', LogisticRegression(solver = 'saga', class_weight = 'balanced', penalty = 'elasticnet', max_iter = 1000,  C = 0.091, tol = 0.1, l1_ratio = 0.9)),
                'Random Forest' :('rf', RandomForestClassifier(class_weight = 'balanced', max_depth = 2, max_features = 'sqrt')),
                'KNearest Neighbors' :('knn', KNeighborsClassifier(n_neighbors = 1, p = 1, weights = 'uniform')),
                'SVC' :('svc', SVC(class_weight = 'balanced', C= 1, gamma= 'scale', probability = True))}


# Canbind Variables