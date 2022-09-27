:: Create X datasets and do initial processesing

:: python code/data-cleaning/stard_preprocessing_manager.py data/stard_data -a
:: python code/data-cleaning/canbind_preprocessing_manager.py
:: python code/data-cleaning/generate_overlapping_features.py -both data/stard_data/processed_data/final_xy_data_matrices data/canbind_data/

:: 4) Create y targets:
:: python code/create_targets.py data/stard_data/processed_data/final_xy_data_matrices/y_qlesq_77__final
:: python code/create_targets.py data/stard_data/processed_data/final_xy_data_matrices/y_qlesq_91__final
:: python code/create_targets.py data/canbind_data/q-les-q/canbind_qlesq_y

:: 5) Move Relevant Datasets to modelling folder:

:: python code/move_modelling.py

:: 6) Create Train/Test Data:

:: python code/train_test_split.py X_77_qlesq_sr__final_extval.csv y_qlesq_77__final__targets.csv
:: python code/train_test_split.py X_77_qlesq_sr__final.csv y_qlesq_77__final__targets.csv

:: 7) Create QIDS/QLESQ subsets

:: python code/qids_qlesq_subprep.py full

:: 8) Elastic Net Selection

:: python code/enet_selection.py full
:: python code/enet_selection.py over
:: python code/enet_selection.py full_noqidsqlesq

:: 8/9) Running an experiment: (can't remember if this is redundant or not...)
:: 9) Run an Evaluation on STARD and CANBIND holdout sets:

python code/run_evaluation.py full JJ_test
