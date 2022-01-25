# # move relevant files
# from pathlib import path
import typer
import datetime
import os
import shutil


def main():

    startTime = datetime.datetime.now()
    modelling_path = "data/modelling"

    if os.path.isdir(modelling_path)  == False:
        os.mkdir(modelling_path + "/")


    # Move STAR*D data
    shutil.move("data/STARD_raw/processed_data/final_xy_data_matrices/y_qlesq_77__final__targets.csv", "data/modelling/y_qlesq_77__final__targets.csv")
    shutil.move("data/STARD_raw/processed_data/final_xy_data_matrices/y_qlesq_91__final__targets.csv", "data/modelling/y_qlesq_91__final__targets.csv")
    shutil.move("data/STARD_raw/processed_data/final_xy_data_matrices/X_77_qlesq_sr__final.csv", "data/modelling/X_77_qlesq_sr__final.csv")
    shutil.move("data/STARD_raw/processed_data/final_xy_data_matrices/X_91_qlesq_sr__final.csv", "data/modelling/X_91_qlesq_sr__final.csv")
    shutil.move("data/STARD_raw/processed_data/final_xy_data_matrices/X_77_qlesq_sr__final_extval.csv", "data/modelling/X_77_qlesq_sr__final_extval.csv")
    shutil.move("data/STARD_raw/processed_data/final_xy_data_matrices/X_91_qlesq_sr__final_extval.csv", "data/modelling/X_91_qlesq_sr__final_extval.csv")


    # Move CAN-BIND
    shutil.move("data/canbind_raw_data/X_test_cb_extval.csv", "data/modelling/X_test_cb_extval.csv")
    shutil.move("data/canbind_raw_data/q-les-q/canbind_qlesq_y__targets.csv", "data/modelling/canbind_qlesq_y__targets.csv")        

    # Path("path/to/current/file.foo").rename("path/to/new/destination/for/file.foo")

    print(f"Completed in: {datetime.datetime.now() - startTime}")

if __name__ == "__main__":
    typer.run(main)