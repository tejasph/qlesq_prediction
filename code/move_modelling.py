# # move relevant files
# from pathlib import path
import typer
import datetime
import os
import shutil


def main():

    # These should be moved to a global file at some point
    stard_folder_name = "stard_data"  # STARD_raw
    canbind_folder_name = "canbind_data"  # canbind_raw_data

    startTime = datetime.datetime.now()
    modelling_path = "data/modelling"

    if os.path.isdir(modelling_path)  == False:
        os.mkdir(modelling_path + "/")

    # Move STAR*D data
    shutil.move(f"data/{stard_folder_name}/processed_data/final_xy_data_matrices/y_qlesq_77__final__targets.csv", "data/modelling/y_qlesq_77__final__targets.csv")
    shutil.move(f"data/{stard_folder_name}/processed_data/final_xy_data_matrices/y_qlesq_91__final__targets.csv", "data/modelling/y_qlesq_91__final__targets.csv")
    
    shutil.move(f"data/{stard_folder_name}/processed_data/final_xy_data_matrices/X_77_qlesq_sr__final.csv", "data/modelling/X_77_qlesq_sr__final.csv")
    shutil.move(f"data/{stard_folder_name}/processed_data/final_xy_data_matrices/X_91_qlesq_sr__final.csv", "data/modelling/X_91_qlesq_sr__final.csv")
    shutil.move(f"data/{stard_folder_name}/processed_data/final_xy_data_matrices/X_77_qlesq_sr__final_extval.csv", "data/modelling/X_77_qlesq_sr__final_extval.csv")
    shutil.move(f"data/{stard_folder_name}/processed_data/final_xy_data_matrices/X_91_qlesq_sr__final_extval.csv", "data/modelling/X_91_qlesq_sr__final_extval.csv")


    # Move CAN-BIND
    shutil.move(f"data/{canbind_folder_name}/X_test_cb_extval.csv", "data/modelling/X_test_cb_extval.csv")
    shutil.move(f"data/{canbind_folder_name}/q-les-q/canbind_qlesq_y__targets.csv", "data/modelling/canbind_qlesq_y__targets.csv")

    # Path("path/to/current/file.foo").rename("path/to/new/destination/for/file.foo")

    print(f"Completed in: {datetime.datetime.now() - startTime}")

if __name__ == "__main__":
    typer.run(main)