# Note: This code is not integrated into the project pipeline. It is here as an example of how to create
# beeswarm plots using shap values.
import pandas as pd
import numpy as np
import re 
import pickle
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScale

X_train = pd.read_csv("qlesq_prediction/data/modelling/X_train_77.csv").set_index("subjectkey")
y_train = pd.read_csv("qlesq_prediction/data/modelling/y_train_77.csv").set_index('subjectkey')
X_test = pd.read_csv("qlesq_prediction/data/modelling/X_test_77.csv").set_index("subjectkey")
y_test = pd.read_csv("qlesq_prediction/data/modelling/y_test_77.csv")
file = open("qlesq_prediction/results/evaluations/test_full_shap/models/Random_Forest_44", 'rb')


# Load a model for an example, which has already been fit on the training data ( in this case, model with avg performance was selected)
rf_model = pickle.load(file)

# Apply scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create tree explainer object
explainer = shap.TreeExplainer(rf_model.steps[1][1])


# shap_values = explainer.shap_values(X_test_scaled)
# create shap object 
shap_obj = explainer(X_test_scaled)

shap.summary_plot(shap_values = np.take(shap_obj.values, 1 , axis = -1), features =  X_test, plot_size = (15,10), max_display = 30)