#globals
# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

DATA_MODELLING_FOLDER = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\data\modelling"
GRID_SEARCH_RESULTS = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\grid_search"
EXPERIMENT_RESULTS = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\experiments"
EVALUATION_RESULTS = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations"


# Excellent article on class imbalance management in GBDT (https://towardsdatascience.com/practical-tips-for-class-imbalance-in-binary-classification-6ee29bcdb8a7)

# Full  Variables
full_feat_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy = 'stratified')), 
      'Random_Forest' : ('rf', RandomForestClassifier(class_weight = 'balanced', max_depth = 2, max_features = 'sqrt')),
      'Logistic_Regression': ('lr', LogisticRegression(solver = 'saga', class_weight = 'balanced', penalty = 'elasticnet', max_iter = 1000,  C = 0.096, tol = 0.1, l1_ratio = 0.8)),
      'KNearest_Neighbors': ('knn', KNeighborsClassifier(n_neighbors = 15, p = 1, weights = 'uniform')),
      'Support_Vector_Machine':('svc', SVC(class_weight = 'balanced', C = 1, gamma = 'auto', probability = True)),
      'Gradient Boosting Classifier': ('gbdt', GradientBoostingClassifier(max_features = 'sqrt'))}




# Overlapping  Variables

overlapping_feat_models = {'Dummy_Classification': ('dummy', DummyClassifier(strategy = 'stratified')),
                'Logistic_Regression': ('lr', LogisticRegression(solver = 'saga', class_weight = 'balanced', penalty = 'elasticnet', max_iter = 1000,  C = 0.091, tol = 0.1, l1_ratio = 0.9)),
                'Random_Forest' :('rf', RandomForestClassifier(class_weight = 'balanced', max_depth = 2, max_features = 'sqrt')),
                'KNearest_Neighbors' :('knn', KNeighborsClassifier(n_neighbors = 1, p = 1, weights = 'uniform')),
                'Support_Vector_Machine' :('svc', SVC(class_weight = 'balanced', C= 1, gamma= 'scale', probability = True))}


# Full Features with RFE (30 features selected)
full_feat_models_rfe = {'Dummy_Classification': ('dummy', DummyClassifier(strategy = 'stratified')),
                'Logistic_Regression': ('lr', LogisticRegression(solver = 'saga', class_weight = 'balanced', penalty = 'elasticnet', max_iter = 1000,  C = 0.105, tol = 0.1, l1_ratio = 0.7)),
                'Random_Forest' :('rf', RandomForestClassifier(class_weight = 'balanced', criterion = 'entropy', max_depth = 2, max_features = 0.1, min_impurity_decrease = 0.0, min_samples_leaf = 5, min_samples_split = 2)),
                'KNearest_Neighbors' :('knn', KNeighborsClassifier(n_neighbors = 7, p = 1, weights = 'uniform')),
                'Support_Vector_Machine' :('svc', SVC(class_weight = 'balanced', C= 1, gamma= 'scale', probability = True))}