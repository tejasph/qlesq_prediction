# stat_testing.py
import pandas as pd
import datetime
import typer
import os
from scipy.stats import ttest_ind
from globals import STAT_RESULTS

def main():

    startTime = datetime.datetime.now()

    if os.path.exists(STAT_RESULTS) == False:
        os.mkdir(STAT_RESULTS)

    output_path = STAT_RESULTS + '/table_1_stat_grid.csv'

    # Table 1 

    # Paths involved
    STARD_OVER_EVAL = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations\STARD_OVER_EVAL"
    CANBIND_EVAL = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations\CANBIND_EVAL"

    stard_full = pd.read_csv(os.path.join(STARD_OVER_EVAL, "STARD_OVER_EVAL_raw_scores.csv"))
    stard_full_enet = pd.read_csv(os.path.join(CANBIND_EVAL, "STARD_CANBIND_EVAL_raw_scores.csv"))

    # cols that we want p-values for
    bal_acc_1 = ["Dummy_Classification_bal_acc", "Logistic_Regression_bal_acc", "Random_Forest_bal_acc",
     "Elastic_Net_bal_acc", "KNearest_Neighbors_bal_acc", "Support_Vector_Machine_bal_acc", "Gradient Boosting Classifier_bal_acc"]

    bal_acc_2 = ["Dummy_Classification_bal_acc", "Logistic_Regression_bal_acc", "Random_Forest_bal_acc",
     "Elastic_Net_bal_acc", "KNearest_Neighbors_bal_acc", "Support_Vector_Machine_bal_acc", "Gradient Boosting Classifier_bal_acc"]

    auc_1 = ["Dummy_Classification_auc", "Logistic_Regression_auc", "Random_Forest_auc",
     "Elastic_Net_auc", "KNearest_Neighbors_auc", "Support_Vector_Machine_auc", "Gradient Boosting Classifier_auc"]

    auc_2 = ["Dummy_Classification_auc", "Logistic_Regression_auc", "Random_Forest_auc",
     "Elastic_Net_auc", "KNearest_Neighbors_auc", "Support_Vector_Machine_auc", "Gradient Boosting Classifier_auc"]


    f = open(output_path, 'w+')
    f.write('T_test Grid\n Balanced Accuracy two-tailed two-sided t-tests\n')
    f.write("STARD EVAL (k = 100) Balanced Accuracy P-Values\n")

    ######################### Compare bal accs
    f.write(" ,")
    for col1 in bal_acc_1:
        f.write(col1 + ",")
    
    for col1 in bal_acc_1:
        f.write("\n" + col1 + ",")
        for col2 in bal_acc_2:
            print(col1)
            print(col2)
            print(ttest_ind(stard_full[col1], stard_full[col2]).pvalue)
            f.write(f'{ttest_ind(stard_full[col1], stard_full[col2]).pvalue},')
    #########################

    f.write("\n\n STARD EVAL (k = 100) AUC P-Values \n\n")
    ######################### Compare AUCS
    f.write(" ,")
    for col1 in auc_1:
        f.write(col1 + ",")
    
    for col1 in auc_1:
        f.write("\n" + col1 + ",")
        for col2 in auc_2:
            print(col1)
            print(col2)
            print(ttest_ind(stard_full[col1], stard_full[col2]).pvalue)
            f.write(f'{ttest_ind(stard_full[col1], stard_full[col2]).pvalue},')
    #########################


    f.write("\n\nCANBIND EVAL (k = 100) Balanced Accuracy P-Values\n\n")

    ######################### Compare bal accs
    f.write(" ,")
    for col1 in bal_acc_1:
        f.write(col1 + ",")
    
    for col1 in bal_acc_1:
        f.write("\n" + col1 + ",")
        for col2 in bal_acc_2:
            print(col1)
            print(col2)
            print(ttest_ind(stard_full_enet[col1], stard_full_enet[col2]).pvalue)
            f.write(f'{ttest_ind(stard_full_enet[col1], stard_full_enet[col2]).pvalue},')
    #########################

    f.write("\n\n CANBIND EVAL (k = 100) AUC P-Values \n\n")

    ######################### Compare AUCS
    f.write(" ,")
    for col1 in auc_1:
        f.write(col1 + ",")
    
    for col1 in auc_1:
        f.write("\n" + col1 + ",")
        for col2 in auc_2:
            print(col1)
            print(col2)
            print(ttest_ind(stard_full_enet[col1], stard_full_enet[col2]).pvalue)
            f.write(f'{ttest_ind(stard_full_enet[col1], stard_full_enet[col2]).pvalue},')
    #########################

    f.write("\n\n STAR*D holdout vs CAN-BIND-1 holdout  \n\n")
    bal_accs = ["Dummy_Classification_bal_acc", "Logistic_Regression_bal_acc", "Random_Forest_bal_acc",
    "KNearest_Neighbors_bal_acc", "Support_Vector_Machine_bal_acc", "Gradient Boosting Classifier_bal_acc"]

    ######################### Model vs Model w/ Enet

    f.write("Metric , Balanced Acc P-Value," )
    for col in bal_acc_1:
        f.write("\n")
        f.write(f'{col},')
        f.write(f'{ttest_ind(stard_full[col], stard_full_enet[col]).pvalue},')

    f.write("\n\nMetric , AUC P-Value," )
    for col in auc_1:
        f.write("\n")
        f.write(f'{col},')
        f.write(f'{ttest_ind(stard_full[col], stard_full_enet[col]).pvalue},')

    ########################
    f.close()

    print(f"Completed in: {datetime.datetime.now() - startTime}")

if __name__ == "__main__":
    typer.run(main)
