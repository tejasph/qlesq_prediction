# Table_2_stats.py
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

    output_path = STAT_RESULTS + '/table_2_stat_grid.csv'

    # Table 2

    # Paths involved
    STARD_FULL_EVAL = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations\STARD_FULL_EVAL"
    STARD_NOQIDSQLESQ_FULL_EVAL =  r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations\STARD_FULL_NOQIDSQLESQ_EVAL"

    STARD_OVER_EVAL = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations\STARD_OVER_EVAL"
    STARD_QIDS_OVER_EVAL =  r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations\STARD_over_qids_EVAL"
    STARD_QLESQ_OVER_EVAL = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations\STARD_over_qlesq_EVAL"
    STARD_QIDSQLESQ_OVER_EVAL =  r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations\STARD_over_qidsqlesq_EVAL"
    STARD_NOQIDSQLESQ_OVER_EVAL =  r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations\STARD_over_noqidsqlesq_EVAL"
    

    CANBIND_EVAL = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations\CANBIND_EVAL"
    CANBIND_QIDS_EVAL = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations\CANBIND_qids_EVAL"
    CANBIND_QLESQ_EVAL = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations\CANBIND_qlesq_EVAL"
    CANBIND_QIDSQLESQ_EVAL = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations\CANBIND_qidsqlesq_EVAL"
    CANBIND_NOQIDSQLESQ_EVAL = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations\CANBIND_noqidsqlesq_EVAL"

    stard_full = pd.read_csv(os.path.join(STARD_FULL_EVAL, "STARD_FULL_EVAL_raw_scores.csv"))
    stard_noqidsqlesq_full_eval = pd.read_csv(os.path.join(STARD_NOQIDSQLESQ_FULL_EVAL, "STARD_FULL_NOQIDSQLESQ_EVAL_raw_scores.csv"))

    #Note: naming conventions of the files depends on how user specified whilst running their evaluation scripts
    stard_over_eval = pd.read_csv(os.path.join(STARD_OVER_EVAL, "STARD_OVER_EVAL_raw_scores.csv"))
    stard_qids_over_eval = pd.read_csv(os.path.join(STARD_QIDS_OVER_EVAL, "STARD_qids_EVAL_raw_scores.csv"))
    stard_qlesq_over_eval = pd.read_csv(os.path.join(STARD_QLESQ_OVER_EVAL, "STARD_qlesq_EVAL_raw_scores.csv"))
    stard_qidsqlesq_over_eval = pd.read_csv(os.path.join(STARD_QIDSQLESQ_OVER_EVAL, "STARD_qidsqlesq_EVAL_raw_scores.csv"))
    stard_noqidsqlesq_over_eval = pd.read_csv(os.path.join(STARD_NOQIDSQLESQ_OVER_EVAL, "STARD_over_noqidsqlesq_EVAL_raw_scores.csv"))

    canbind_eval = pd.read_csv(os.path.join(CANBIND_EVAL, "STARD_CANBIND_EVAL_raw_scores.csv"))
    canbind_qids_eval = pd.read_csv(os.path.join(CANBIND_QIDS_EVAL, "CANBIND_QIDS_EVAL_raw_scores.csv"))
    canbind_qlesq_eval = pd.read_csv(os.path.join(CANBIND_QLESQ_EVAL, "CANBIND_QLESQ_EVAL_raw_scores.csv"))
    canbind_qidsqlesq_eval = pd.read_csv(os.path.join(CANBIND_QIDSQLESQ_EVAL, "CANBIND_QIDSQLESQ_EVAL_raw_scores.csv"))
    canbind_noqidsqlesq_eval = pd.read_csv(os.path.join(CANBIND_NOQIDSQLESQ_EVAL, "CANBIND_NOQIDSQLESQ_EVAL_raw_scores.csv"))

    # cols that we want p-values for
    # bal_acc_1 = ["Dummy_Classification_bal_acc", "Logistic_Regression_bal_acc", "Random_Forest_bal_acc",
    #  "Elastic_Net_bal_acc", "KNearest_Neighbors_bal_acc", "Support_Vector_Machine_bal_acc", "Gradient Boosting Classifier_bal_acc"]

    # bal_acc_2 = ["Dummy_Classification_bal_acc", "Logistic_Regression_bal_acc", "Random_Forest_bal_acc",
    #  "Elastic_Net_bal_acc", "KNearest_Neighbors_bal_acc", "Support_Vector_Machine_bal_acc", "Gradient Boosting Classifier_bal_acc"]

    # auc_1 = ["Dummy_Classification_auc", "Logistic_Regression_auc", "Random_Forest_auc",
    #  "Elastic_Net_auc", "KNearest_Neighbors_auc", "Support_Vector_Machine_auc", "Gradient Boosting Classifier_auc"]

    # auc_2 = ["Dummy_Classification_auc", "Logistic_Regression_auc", "Random_Forest_auc",
    #  "Elastic_Net_auc", "KNearest_Neighbors_auc", "Support_Vector_Machine_auc", "Gradient Boosting Classifier_auc"]

    dfs_1 = [stard_full, stard_noqidsqlesq_full_eval, stard_over_eval, stard_qids_over_eval, stard_qlesq_over_eval, stard_qidsqlesq_over_eval, stard_noqidsqlesq_over_eval]
    dfs_2 = [stard_full, stard_noqidsqlesq_full_eval, stard_over_eval, stard_qids_over_eval, stard_qlesq_over_eval, stard_qidsqlesq_over_eval, stard_noqidsqlesq_over_eval]
    names = ['stard_full', 'stard_noqidsqlesq_full_eval', 'stard_over_eval', 'stard_qids_over_eval', 'stard_qlesq_over_eval', 'stard_qidsqlesq_over_eval', 'stard_noqidsqlesq_over_eval']

    f = open(output_path, 'w+')
    f.write('T_test Grid\n Balanced Accuracy two-tailed two-sided t-tests\n')
    f.write("Overlapping STAR*D (n = 100): Balanced Accuracy and AUC P-Values\n")

    ######################### Overlapping STAR*D various feature sets

    f.write("Metric , Balanced Acc P-Value,\n ," )
    name_index = 0
    for x in dfs_1:
        f.write(names[name_index] + ",")
        name_index += 1

    name_index = 0 
    for x in dfs_1:
        f.write("\n")
        f.write(f'{names[name_index]},')
        for y in dfs_2:
            print(x['Random_Forest_bal_acc'])
            print(y['Random_Forest_bal_acc'])
            print(ttest_ind(x['Random_Forest_bal_acc'], y['Random_Forest_bal_acc']).pvalue)
            p_val = ttest_ind(x['Random_Forest_bal_acc'], y['Random_Forest_bal_acc']).pvalue
            f.write(str(p_val) + ",")
        name_index += 1

    f.write("\n\nMetric , AUC P-Value, \n ," )

    name_index = 0
    for x in dfs_1:
        f.write(names[name_index] + ",")
        name_index += 1

    name_index = 0 
    for x in dfs_1:
        f.write("\n")
        f.write(f'{names[name_index]},')
        for y in dfs_2:
            print(x['Random_Forest_auc'])
            print(y['Random_Forest_auc'])
            print(ttest_ind(x['Random_Forest_auc'], y['Random_Forest_auc']).pvalue)
            p_val = ttest_ind(x['Random_Forest_auc'], y['Random_Forest_auc']).pvalue
            f.write(str(p_val) + ",")
        name_index += 1

    ########################

    # Remove enet model for next comparisons
    # bal_acc_1.remove("Elastic_Net_bal_acc")
    # bal_acc_2.remove("Elastic_Net_bal_acc")
    # auc_1.remove("Elastic_Net_auc")
    # auc_2.remove("Elastic_Net_auc")

    canbind_1 = [canbind_eval, canbind_qids_eval, canbind_qlesq_eval, canbind_qidsqlesq_eval, canbind_noqidsqlesq_eval]
    canbind_2 =  [canbind_eval, canbind_qids_eval, canbind_qlesq_eval, canbind_qidsqlesq_eval, canbind_noqidsqlesq_eval]
    canbind_names = ['canbind_eval', 'canbind_qids_eval', 'canbind_qlesq_eval', 'canbind_qidsqlesq_eval', 'canbind_noqidsqlesq_eval']

    f.write("\n\n CANBIND  Balanced Accuracy and AUC P-Values\n\n")

    f.write("Metric , Balanced Acc P-Value,\n ," )
    name_index = 0
    for x in canbind_1:
        f.write(canbind_names[name_index] + ",")
        name_index += 1

    name_index = 0 
    for x in canbind_1:
        f.write("\n")
        f.write(f'{canbind_names[name_index]},')
        for y in canbind_2:
            print(x['Random_Forest_bal_acc'])
            print(y['Random_Forest_bal_acc'])
            print(ttest_ind(x['Random_Forest_bal_acc'], y['Random_Forest_bal_acc']).pvalue)
            p_val = ttest_ind(x['Random_Forest_bal_acc'], y['Random_Forest_bal_acc']).pvalue
            f.write(str(p_val) + ",")
        name_index += 1

    f.write("\n\nMetric , AUC P-Value, \n ," )

    name_index = 0
    for x in canbind_1:
        f.write(canbind_names[name_index] + ",")
        name_index += 1

    name_index = 0 
    for x in canbind_1:
        f.write("\n")
        f.write(f'{canbind_names[name_index]},')
        for y in canbind_2:
            print(x['Random_Forest_auc'])
            print(y['Random_Forest_auc'])
            print(ttest_ind(x['Random_Forest_auc'], y['Random_Forest_auc']).pvalue)
            p_val = ttest_ind(x['Random_Forest_auc'], y['Random_Forest_auc']).pvalue
            f.write(str(p_val) + ",")
        name_index += 1

    ######################### Overlapping STAR*D  vs CANBIND Balanced Accuracy

    f.write("\n\n Feat Set , Balanced Acc P-Value, \n" )
    f.write("Overlapping (k = 100) ,")
    f.write(str(ttest_ind(stard_over_eval['Random_Forest_bal_acc'], canbind_eval['Random_Forest_bal_acc']).pvalue))
    f.write("\n")

    f.write("Qids (k = 47) ,")
    f.write(str(ttest_ind(stard_qids_over_eval['Random_Forest_bal_acc'], canbind_qids_eval['Random_Forest_bal_acc']).pvalue))
    f.write("\n")

    f.write("Qlesq (k = 16) ,")
    f.write(str(ttest_ind(stard_qlesq_over_eval['Random_Forest_bal_acc'], canbind_qlesq_eval['Random_Forest_bal_acc']).pvalue))
    f.write("\n")

    f.write("Qids + qlesq (k = 63) ,")
    f.write(str(ttest_ind(stard_qidsqlesq_over_eval['Random_Forest_bal_acc'], canbind_qidsqlesq_eval['Random_Forest_bal_acc']).pvalue))
    f.write("\n")

    f.write("No Qids + qlesq (k = 37) ,")
    f.write(str(ttest_ind(stard_noqidsqlesq_over_eval['Random_Forest_bal_acc'], canbind_noqidsqlesq_eval['Random_Forest_bal_acc']).pvalue))
    f.write("\n\n")
    ######################### Overlapping STAR*D  vs CANBIND AUC
    f.write("\n Feat Set , AUC P-Value, \n" )
    f.write("Overlapping (k = 100) ,")
    f.write(str(ttest_ind(stard_over_eval['Random_Forest_auc'], canbind_eval['Random_Forest_auc']).pvalue))
    f.write("\n")

    f.write("Qids (k = 47) ,")
    f.write(str(ttest_ind(stard_qids_over_eval['Random_Forest_auc'], canbind_qids_eval['Random_Forest_auc']).pvalue))
    f.write("\n")

    f.write("Qlesq (k = 16) ,")
    f.write(str(ttest_ind(stard_qlesq_over_eval['Random_Forest_auc'], canbind_qlesq_eval['Random_Forest_auc']).pvalue))
    f.write("\n")

    f.write("Qids + qlesq (k = 63) ,")
    f.write(str(ttest_ind(stard_qidsqlesq_over_eval['Random_Forest_auc'], canbind_qidsqlesq_eval['Random_Forest_auc']).pvalue))
    f.write("\n")

    f.write("No Qids + qlesq (k = 37) ,")
    f.write(str(ttest_ind(stard_noqidsqlesq_over_eval['Random_Forest_auc'], canbind_noqidsqlesq_eval['Random_Forest_auc']).pvalue))
    f.write("\n\n")

    # for col in bal_acc_1:
    #     f.write("\n")
    #     f.write(f'{col},')
    #     f.write(f'{ttest_ind(stard_over_enet[col], canbind_enet[col]).pvalue},')

    # f.write("\n\nMetric , AUC P-Value," )
    # for col in auc_1:
    #     f.write("\n")
    #     f.write(f'{col},')
    #     f.write(f'{ttest_ind(stard_over_enet[col], canbind_enet[col]).pvalue},')

    ########################


    f.close()

    print(f"Completed in: {datetime.datetime.now() - startTime}")

if __name__ == "__main__":
    typer.run(main)
