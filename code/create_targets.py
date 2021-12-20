# create_targets.py
# December 20 2021


import typer
import pandas as pd
import numpy as np
import datetime


def main(y_path: str):

    startTime = datetime.datetime.now()

    df = pd.read_csv(y_path + ".csv")

    df['qlesq_QoL_threshold'] = np.where(df['end_qlesq'] >= 67, 1, 0)
    df['qlesq_change'] = df['end_qlesq'] - df['start_qlesq']
    df['qlesq_resp'] = np.where(df['qlesq_change'] >= 10, 1, 0)

    df.to_csv(y_path + "__targets.csv", index = False)

    print(f"Completed in: {datetime.datetime.now() - startTime}")

if __name__ == "__main__":
    typer.run(main)