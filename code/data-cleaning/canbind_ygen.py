import os
import csv
import pandas as pd
import numpy as np
import sys

from canbind_globals import COL_NAME_PATIENT_ID,COL_NAME_EVENTNAME, COL_NAME_GROUP, GROUP_WHITELIST, YGEN_EVENTNAME_WHITELIST,  TARGET_MAP, YGEN_COL_NAMES_TO_CONVERT
from canbind_globals import YGEN_SCALE_FILENAMES, COL_NAMES_BLACKLIST_COMMON, COL_NAMES_BLACKLIST_QIDS
from canbind_utils import get_event_based_value, aggregate_rows
from canbind_utils import is_number, replace_target_col_values, collect_columns_to_extend
from utils import get_valid_subjects
""" 
Generates a y-matrix from the CAN-BIND data. Similiar code to canbind_data_processor, separated to ensure no week8 contamination to data matrix

Example usages

    Basic:
        python canbind_ygen.py /path/to/data/folders

This will output a single CSV file containing the y-matrix

The method expects CSV files to be contained within their own subdirectories from the root directory, as is organized
in the ZIP provided.

TODO: took out most of the superflous code from canbind_preprocessing_manager, which this was based on
      Runs fast, but probably could still take out more and be further optimized. 
"""
def ygen(root_dir, debug=False):
    global COL_NAMES_CATEGORICAL
    global COL_NAMES_NA
    global FILENAMES
    global NUM_DATA_FILES
    global NUM_DATA_ROWS
    global NUM_DATA_COLUMNS

    col_names_categorical = {}
    col_names_na = {}

    filenames = []

    num_data_files = 0
    num_data_rows = 0
    num_data_columns = 0

    merged_df = pd.DataFrame([])

    for subdir, dirs, files in os.walk(root_dir):
        for filename in files:
            file_path = os.path.join(subdir, filename)

            if filename in YGEN_SCALE_FILENAMES:
                filenames.append(filename)
                num_data_files += 1
                
                # Track counts and column names for sanity check
                with open(file_path, 'rt') as csvfile:
                    col_names = []
                    csv_reader = csv.reader(csvfile)
                    for i, row in enumerate(csv_reader):
                        num_data_rows += 1

                        # Store the column names
                        if i == 0:
                            col_names = row
                            num_data_columns += len(row)
                            for field in row:
                                field = field.upper()

                                # Collect names of columns that will be extended with extra columns based on event value
                                collect_columns_to_extend(field)
                        else:
                            # Determine all columns with categorical values
                            for j, field_value in enumerate(row):
                                col_name = col_names[j].upper()
                                if field_value == "":
                                    continue
                                elif is_number(field_value):
                                    continue
                                elif field_value == "NA":
                                    col_names_na[col_name] = True
                                else:
                                    col_names_categorical[col_name] = True

                csvfile.close()

                df = pd.read_csv(file_path)

                # Convert all column names to upper case to standardize names
                df.rename(columns=lambda x: x.upper(), inplace=True)

                # Append the CSV dataframe
                merged_df = merged_df.append(df, sort=False)

    # Sort the rows by the patient identifier
    merged_df = merged_df.sort_values(by=[COL_NAME_PATIENT_ID])

    # Back up full merged file for debugging purposes
    if debug: merged_df.to_csv(root_dir + "/merged-data.unprocessed_ygen.csv")

    #### FILTER ROWS AND COLUMNS ####

    # Filter out rows that are controls
    if COL_NAME_GROUP in merged_df:
        merged_df = merged_df.loc[~merged_df.GROUP.str.lower().isin(GROUP_WHITELIST)]
    
    # Filter out rows that were recorded beyond Week 8
    if COL_NAME_EVENTNAME in merged_df:
        merged_df = merged_df.loc[merged_df.EVENTNAME.str.lower().isin(YGEN_EVENTNAME_WHITELIST)]
        

    #### CREATE NEW COLUMNS AND MERGE ROWS ####

    # Handle column extension based on EVENTNAME or VISITSTATUS
    merged_df, extension_blacklist = extend_columns_eventbased(merged_df)

    # Collapse/merge patient rows
    merged_df = aggregate_rows(merged_df)

    # First replace empty strings with np.nan, as pandas knows to ignore creating one-hot columns for np.nan
    # This step is necessary for one-hot encoding and for replacing nan values with a median
    merged_df = merged_df.replace({"": np.nan})

    # Finalize the blacklist, then do a final drop of columns (original ones before one-hot and blacklist columns)
    blacklist_ygen = COL_NAMES_BLACKLIST_QIDS
    blacklist_ygen.extend(COL_NAMES_BLACKLIST_COMMON)
    blacklist_ygen.extend(extension_blacklist)
    merged_df.drop(blacklist_ygen, axis=1, inplace=True)

    # Create y target, eliminate invalid subjects in both X and y (those who don't make it to week 8), convert responder/nonresponder string to binary
    merged_df = get_valid_subjects(merged_df)
    merged_df = merged_df.drop(["RESPOND_WK8"], axis=1)
    merged_df = replace_target_col_values(merged_df, [TARGET_MAP])
    merged_df = merged_df.sort_values(by=[COL_NAME_PATIENT_ID])
    merged_df = merged_df.reset_index(drop=True)
    
    # Rename the column that will be used for the y value (target)
    merged_df = merged_df.rename({"QIDS_RESP_WK8_week 8":"QIDS_RESP_WK8"},axis='columns',errors='raise')
    merged_df['QIDS_REM_WK8'] = np.nan
    
    # Back up proceesed file before ygeneration
    if debug: merged_df.to_csv(root_dir + "/merged-data_processed_ygen.csv")
    
    canbind_y_mag = pd.DataFrame()
    # Replace missing "QIDS_RESP_WK8" values by manually checking criteria
    for i, row in merged_df.iterrows():
        
        baseline_qids_sr = row['QIDS_OVERL_SEVTY_baseline']
        # week2_qids_sr = row['QIDS_OVERL_SEVTY_week 2']
        week4_qids_sr = row['QIDS_OVERL_SEVTY_week 4']
        week8_qids_sr = row['QIDS_OVERL_SEVTY_week 8']

        # Find LOCF, either week 8 or week 4
        if not(np.isnan(week8_qids_sr)):
            locf_qids_sr = week8_qids_sr
        elif not(np.isnan(week4_qids_sr)):
            locf_qids_sr = week4_qids_sr
        else:
            # If patient does not have a week 4 or 8 qids_sr, do not generate y, they will be dropped
            continue
        
        # Make qids-sr remission at 8 weeks from scratch
        if locf_qids_sr <= 5:
            merged_df.at[i, 'QIDS_REM_WK8'] = 1
        else:
            merged_df.at[i, 'QIDS_REM_WK8'] = 0  
            
        # Fill in any missing qids-sr response at 8 weeks
        if "QIDS_RESP_WK8" in row:
            if np.isnan(row["QIDS_RESP_WK8"]):
                if locf_qids_sr <= baseline_qids_sr*0.50:
                    merged_df.at[i, 'QIDS_RESP_WK8'] = 1
                else:
                    merged_df.at[i, 'QIDS_RESP_WK8'] = 0
            else:
                if locf_qids_sr <= baseline_qids_sr*0.50:
                    assert merged_df.at[i, 'QIDS_RESP_WK8'] == 1, "Found an error when manually checking QIDS_RESP_WK8"
                else:
                    assert merged_df.at[i, 'QIDS_RESP_WK8'] == 0, "Found an error when manually checking QIDS_RESP_WK8"

        # Get target_change and final_score targets
        diff = locf_qids_sr - baseline_qids_sr
        # for week_score in [week2_qids_sr, week4_qids_sr, week8_qids_sr]:
        #     diff = week_score - baseline_qids_sr
        #     if abs(diff) > abs(max_diff):
        #         max_diff = diff     

        
        canbind_y_mag.loc[i, 'subjectkey'] = row['SUBJLABEL']
        canbind_y_mag.loc[i,'baseline'] = baseline_qids_sr
        canbind_y_mag.loc[i, 'target_change'] = diff
        canbind_y_mag.loc[i, 'target_score'] = locf_qids_sr
        
                        
    print(merged_df)          
    
    y_wk8_resp = merged_df[['SUBJLABEL','QIDS_RESP_WK8']]
    y_wk8_resp.to_csv(root_dir + "/y_wk8_resp_canbind.csv", index=False)
    
    y_wk8_rem = merged_df[['SUBJLABEL','QIDS_REM_WK8']]
    y_wk8_rem.to_csv(root_dir + "/y_wk8_rem_canbind.csv", index=False)

    #temp
    merged_df.to_csv(root_dir + "/merged_y.csv", index=False)

    canbind_y_mag.to_csv(root_dir + "/canbind_y_mag.csv", index=False)
    
    # Save the version containing NaN values just for debugging, not otherwise used
    if debug: merged_df.to_csv(root_dir + "/canbind-clean-aggregated-data.with-id.contains-blanks-ygen.csv")

class qlesq_subject_selector():

    def __init__(self, df):
        self.df = df.copy()
        
    def get_random_subject(self):
        
        random_id = list(self.df.SUBJLABEL.sample(1))[0]
        return self.df[self.df['SUBJLABEL'] == random_id]
    
    def get_specific_subject(self, chosen_id):
        
        return self.df[self.df['SUBJLABEL'] == chosen_id]
    
    def filter_treatment_group(self, trt_group = 'Treatment'):
        self.df = self.df[self.df['GROUP'] == trt_group]
        
    def filter_NA(self, subset_cols = ['QLESQA_Tot', 'QLESQB_Tot']):
        # Drop NA values
        self.df = self.df.dropna(subset = subset_cols, how = 'all')

    def filter_min_entries(self):
        
        group = self.df.groupby("SUBJLABEL")
        relevant_ids = []
        for id, data in group:
            
            #If no baseline detected, then not viable candidate
            if "Baseline" in data['EVENTNAME'].values:
                if "Week 8" in data['EVENTNAME'].values:
                    relevant_ids.append(id)
        
        self.df = self.df[self.df['SUBJLABEL'].isin(relevant_ids)]
    
    # def merge_QLESQ_AB(self):

    #     # Fill nas with 0 and then simply add to retain qlesq total (note: this is kind of hacky)
    #     self.df = self.df.fillna(value = {'QLESQA_Tot': 0, 'QLESQB_Tot': 0})
    #     self.df['total_QLESQ'] = self.df['QLESQA_Tot'] + self.df['QLESQB_Tot'] 
    #     self.df = self.df.drop(columns = ['QLESQA_Tot', 'QLESQB_Tot'])
    
    def get_relevant_ids(self):
        print("working")
        group = self.df.groupby('SUBJLABEL')
        relevant_ids = []
        for id, data in group:
            
            # If no baseline detected, then not viable candidate
            if "Baseline" in data['EVENTNAME'].values:
                if "Week 8" in data['EVENTNAME'].values:
    
                    relevant_ids.append(id)

        print(f"Number of ids that fit criteria: {len(relevant_ids)}")
        return relevant_ids

def generate_col_names(root, i = 14):
    '''
    Helper function to create column names
    '''
    col_list = []
    for num in range(1, i + 1):
        col_list.append(root + str(num))
        
    return col_list
# def check_qlesq_criteria(df):
#     group_df = df.groupby('SUBJLABEL')
#     for subject, group in group_df:

# #         group = group.sort_values(by = ['days_baseline'], ascending = True)

# #         baseline = group.iloc[0]['totqlesq']
# #         end_score = group.iloc[-1]['totqlesq']]


#         assert group['QLESQA_Tot'].isna().sum() == 0, f"Total Qlesq has {group['QLESQA_Tot'].isna().sum()} NA values for {subject}"
#         assert "Baseline" in group['EVENTNAME'].values, f"No Baseline found for {subject}"
#         assert "Week 8" in group['EVENTNAME'].values, f"No Week 8 entry fround for {subject}"
# #         assert group.duplicated().sum() == 0, f"Duplicate rows detected for {subject}"
# #         assert group.shape[0] >= 2, f"Subject profile has 1 row or less, for {subject}"
          
def qlesq_y_gen(root_dir):

    can_qlesq = pd.read_csv(root_dir  + "CBN01_QLESQ_DATA_forREVEIW.csv")

    # Keep useful columns
    cols_to_drop = ['SITESYMBOL', 'Cohort_ID', 'AGE', 'SEX','VISITDATE', 'Dosage_ID']
    can_qlesq = can_qlesq.drop(columns = cols_to_drop)

    selector = qlesq_subject_selector(can_qlesq)
    print(len(selector.df.SUBJLABEL.unique()))

    # Only keep treatment group
    selector.filter_treatment_group()
    print(len(selector.df.SUBJLABEL.unique()))

    # Remove rows with NA for total QLESQ score (In Both A and B columns)
    selector.filter_NA()
    print(len(selector.df.SUBJLABEL.unique()))

    # Only keep ids where a Baseline and a Week 8 score are present
    selector.filter_min_entries()
    print(len(selector.df.SUBJLABEL.unique()))

    # We expect 176 valid entries for CanBind as per checks in Jupyter Lab
    
    # Generate some column names that correspond to raw excel file
    A_columns = generate_col_names("QLESQ_1A_")
    B_columns = generate_col_names("QLESQ_1B_")

    #Iterate over each row and gather QLESQ values
    max_raw_total = []
    actual_raw_total = []
    transformed_total = []
    for index, row in selector.df.iterrows():
        # A_column path
        if row['QLESQ_0'] == 'Y':
            answer_counter = 0
            for col in A_columns:
                # Check if person answer all questions --> calculate max possible score
                if np.isnan(row[col]) == False:
                    answer_counter += 1

            max_raw = answer_counter * 5
            actual_raw = row['QLESQA_Tot']

        #B column path
        if row['QLESQ_0'] == 'N':
            answer_counter = 0
            for col in B_columns:
                # Check if person answer all questions --> calculate max possible score
                if np.isnan(row[col]) == False:
                    answer_counter += 1

            max_raw = answer_counter * 5
            actual_raw = row['QLESQB_Tot']

        max_raw_total.append(max_raw)
        actual_raw_total.append(actual_raw)
        transformed_total.append(round((actual_raw - answer_counter)/(max_raw - answer_counter)*100, 1))

    # Merge A and B totals to get one communal QLESQ total column --> 'total_QLESQ'
    # selector.merge_QLESQ_AB()
    selector.df['max_raw_total'] = max_raw_total
    selector.df['total_raw_QLESQ'] = actual_raw_total
    selector.df['transformed_qlesq'] = transformed_total

    filtered_df = selector.df.copy()

    group_df = filtered_df.groupby('SUBJLABEL')
    qlesq_y = pd.DataFrame()
    counter = {'week8':0, 'baseline':0, 'threshold':0}
    i = 0
    for subject, group in group_df:


        if "Week 8" not in group['EVENTNAME'].values:
            counter['week8'] += 1
            continue

        if "Baseline" not in group['EVENTNAME'].values:
            counter['baseline'] += 1
            continue


        assert group.shape[0] <= 3, f"Shouldn't be more than 4 rows for {subject}"
        assert group.duplicated().sum() == 0, f"Duplicate rows detected for {subject}"
        assert group['transformed_qlesq'].isna().sum() == 0, f"Total Qlesq has {group['transformed_qlesq'].isna().sum()} NA values for {subject}"
        assert "Baseline" in group['EVENTNAME'].values, f"No Baseline found for {subject}"
        assert "Week 8" in group['EVENTNAME'].values, f"No Week 8 entry fround for {subject}"

        # if "Baseline" not in group['EVENTNAME'].values:
        #     baseline = "NA"
        # else:
        baseline = group[group['EVENTNAME'] == 'Baseline']['transformed_qlesq'].values[0]
        baseline_max_raw = group[group['EVENTNAME'] == 'Baseline']['max_raw_total'].values[0]
        baseline_actual_raw = group[group['EVENTNAME'] == 'Baseline']['total_raw_QLESQ'].values[0]

        
        # Exclude patients who started with a Q-LES-Q baseline within community norm (ie. greater than 66)
        if baseline >= 67:
            counter['threshold'] +=1
            continue

        assert baseline < 67, f"Patient {subject} starting with baseline Quality of Life that is already within 1SD of community norm (>= 67)"
            
        week8 = group[group['EVENTNAME'] == 'Week 8']['transformed_qlesq'].values[0]
        week8_max_raw = group[group['EVENTNAME'] == 'Week 8']['max_raw_total'].values[0]
        week8_actual_raw = group[group['EVENTNAME'] == 'Week 8']['total_raw_QLESQ'].values[0]

        qlesq_y.loc[i, 'subjectkey'] = subject
        qlesq_y.loc[i, 'start_qlesq'] = baseline
        qlesq_y.loc[i, 'baseline_max_raw'] = baseline_max_raw
        qlesq_y.loc[i, 'baseline_actual_raw'] = baseline_actual_raw
        qlesq_y.loc[i, 'end_qlesq'] = week8
        qlesq_y.loc[i, 'week8_max_raw'] = week8_max_raw
        qlesq_y.loc[i, 'week8_actual_raw'] = week8_actual_raw
        i += 1 

    # Rename columns for future script compatability
 
    print("writing qlesq_y")
    print(counter)
    print(len(qlesq_y.subjectkey.unique()))
    qlesq_y.to_csv(root_dir + "canbind_qlesq_y.csv", index = False)

    


def extend_columns_eventbased(orig_df):
    """
    Handles adding extra columns based on a condition the value of another column.

    :param orig_df: the original dataframe
    :return: a new, modified dataframe
    """
    global COL_NAMES_NEW_FROM_EXTENSION
    extension_blacklist = []
        
    # Create extra columns with name of event appended, initialized blank
    for scale_group in YGEN_COL_NAMES_TO_CONVERT:
        scale_name = scale_group[0]
        scale_events_whitelist = scale_group[1]
        col_names = scale_group[2]

        for col_name in col_names:
            for event in scale_events_whitelist:
                # Only add extra columns for entries with a non-empty and valid event
                if type(event) != type("") or is_number(event):
                    continue

                new_col_name = col_name + "_" + event

                # Set the value for the new column
                orig_df[new_col_name] = orig_df.apply(lambda row: get_event_based_value(row, event, col_name, scale_name), axis=1)

        extension_blacklist.extend(col_names)

    return orig_df, extension_blacklist


if __name__ == "__main__":
    if len(sys.argv) == 1:
        pathData = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\canbind_data\\'
        ygen(pathData, debug=False)
    elif len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        ygen(sys.argv[1])
    else:
        print("Enter valid arguments\n"
               "\t path: the path to a real directory\n")
