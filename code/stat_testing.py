# stat_testing.py
import pandas as pd
import datetime
import typer
import os
from scipy.stats import ttest_ind

def main():

    startTime = datetime.datetime.now()

    # Table 1 

    # Paths involved
    STARD_FULL = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations\STARD_FULL_EVAL"
    STARD_FULL_ENET_EVAL = r"C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\results\evaluations\STARD_FULL_ENET_EVAL"

    stard_full = pd.read_csv(os.path.join(STARD_FULL, "STARD_FULL_EVAL_raw_scores.csv"))
    stard_full_enet = pd.read_csv(os.path.join(STARD_FULL_ENET_EVAL, "STARD_FULL_ENET_EVAL_raw_scores.csv"))

    # pairs we want p values for
    pairs = ["Logistic_Regression_bal_acc", "Random_Forest_bal_acc"]

    for p in pairs:
        print(p)
        print(ttest_ind(stard_full[p], stard_full_enet[p]))

    print(f"Completed in: {datetime.datetime.now() - startTime}")

if __name__ == "__main__":
    typer.run(main)
