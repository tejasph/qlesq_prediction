# Quality Of Life Prediction Project using STAR*D and CAN-BIND-1 data
Tejas Phaterpekar

## Pipeline

Listed below is a step-by-step rundown the project pipeline. All python commands are meant to be run from the root of the repository. 

1) Processing the STARD data:

    - `python code/data-cleaning/stard_preprocessing_manager.py data/STARD_raw -a`

2) Processing the CANBIND data:

    - `python code/data-cleaning/canbind_preprocessing_manager.py`

    **note**: before running this, go into canbind_preprocessing_manager.py and alter pathData to a correct pathing for the raw data (see below; will be different depending on the user)

        elif len(sys.argv) == 1:
            pathData = r'C:\Users\Tejas\Documents\qlesq_project\qlesq_prediction\data\canbind_raw_data'
            aggregate_and_clean(pathData, verbose=False)
            ygen(pathData)
            impute(pathData)
            convert_canbind_to_overlapping(pathData)


3) Create Overlapping Data from processed data:

    - `python code/data-cleaning/generate_overlapping_features.py -both data/STARD_raw/processed_data/final_xy_data_matrices data/canbind_raw_data/`

4) Create y targets:

    - `python code/create_targets.py data/STARD_raw/processed_data/final_xy_data_matrices/y_qlesq_77__final`

    - `python code/create_targets.py data/STARD_raw/processed_data/final_xy_data_matrices/y_qlesq_91__final` # redundant now

    - `python code/create_targets.py data/canbind_raw_data/q-les-q/canbind_qlesq_y`

5) Move Relevant Datasets to modelling folder:

    - `python code/move_modelling.py`

6) Create Train/Test Data:

    - `python code/train_test_split.py X_77_qlesq_sr__final_extval.csv y_qlesq_77__final__targets.csv `

    - `python code/train_test_split.py X_77_qlesq_sr__final.csv y_qlesq_77__final__targets.csv` 

7) Create QIDS/QLESQ subsets

    - `python code/qids_qlesq_subprep.py full`
    - `python code/qids_qlesq_subprep.py over`

8) Elastic Net Selection

- `python code/enet_selection.py [eval_type]`

    4 options for eval_type: 
- Run feat selection on full dataset  (eval_type = "full")
- Run feat selectoin on overlapping dataset (eval_type = "over")
- Run feat selection on full dataset with no qidsqlesq feats (eval_type = "full_noqidsqlesq) [not used in paper]
- Run feat selection on overlapping dataset with no qidsqlesq feats (eval_type = "over_noqidsqlesq) [not used in paper]



9) Run an Evaluation on STARD and CANBIND holdout sets:

['full', 'full_enet',  'over', 'over_enet', 'canbind', 'canbind_enet', 'full_qids', 'full_qlesq', 'full_qidsqlesq', 'full_noqidsqlesq', 'full_noqidsqlesq_enet',
                        'over_qids', 'over_qlesq', 'over_qidsqlesq', 'over_noqidsqlesq', 'canbind_qids', 'canbind_qlesq', 'canbind_qidsqlesq', 'canbind_noqidsqlesq'], "eval_type (1st argument) was not valid. The only 3 options are 'full', 'over', and 'canbind'."


    There are several options:
    - Full feature evalution on STARD holdout   (eval_type ="full")
    - Full fetaures w/ enet selection (eval_type = "full_enet")
    - Full dataset with only QIDS features (eval_type = "full_qids")
    - Full dataset with only QLESQ features (eval_type = "full_qlesq")
    - Full dataset with only QIDSQLESQ features (eval_type = "full_qidsqlesq")
    - Full dataset excluding QIDSQLESQ features (eval_type = "full_noqidsqlesq")
    - Full dataset excluding QIDSQLESQ features w/ enet (eval_type = "full_noqidsqlesq_enet") [not used in paper]

    - Overlapping feature evaluation on STARD holdout (eval_type = "over")
    - Overlapping dataset with only QIDS features (eval_type = "over_qids")
    - Overlapping dataset with only QLESQ features (eval_type = "over_qlesq")
    - Overlapping dataset with only QIDSQLESQ features (eval_type = "over_qidsqlesq")
    - Overlapping dataset excluding QIDSQLESQ features (eval_type = "over_noqidsqlesq")

    - Overlapping feature evaluation on CANBIND holdout (eval_type = "canbind")
    - Overlapping features w/ enet selection, evaluated on CANBIND dataset (eval_type = "canbind_enet")
    - Overlapping features with only QIDS features, evaluated on CANBIND (eval_type = "canbind_qids)
    - Overlapping features with only QLESQ features, evaluated on CANBIND (eval_type = "canbind_qlesq)
    - Overlapping features with only QIDSQLESQ features, evaluated on CANBIND (eval_type = "canbind_qidsqlesq)
    - Overlapping features excluding QIDSQLESQ features, evaluated on CANBIND (eval_type = "canbind_noqidsqlesq)

    `python code/run_evaluation.py [eval_type] [a name for the evaluation]`


Optional Scripts:

a) RFE Selection

There are two options for eval type:
- Full features   (eval_type ="full")

- Overlapping features  (eval_type = "over")


Usage: `python code/rfe_selection.py [eval_type] [rfe_step] [n_feats to select]`

ex. `python code/rfe_selection.py full 1 30`

b) Elastic Net Selection

- `python code/enet_selection.py [eval_type]`

2 options for eval_type: 
- Run feat selection on full dataset  (eval_type = "full")
- Run feat selectoin on overlapping dataset (eval_type = "over")

c)  Running a GridSearch (before running, check parameter grids and adjust parameter space as needed)

optimized models are stored under `global.py`


    - `python code/grid_search.py [model_type] [data_type] `

    model_type has several options:
    - rf --> random forest
    - lr --> logistic regression
    - gbdt --> gradient boosting classifiers
    - svc  --> support vector classifier
    - knn --> k nearest neighbors

    data_type has 3 options:
    - full --> full feature set (480)
    - over --> overlapping feature set (100)
    - rfe --> recursive feature elimination set
    - enet --> elastic net CV feature selection

d)8) Running an experiment: (can't remember if this is redundant or not...)

Experiments are run via run_experiment.py. It utilizes classes so code in main() will need to be altered to contain models of your choice. The type of data is given as arguements to main().

