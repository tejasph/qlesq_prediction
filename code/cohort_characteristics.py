import os
import sys
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest


# Analyses the data matrices to produce characteristics of the study cohorts
# Run supplying directory with the X and y matrices for the ext validation portion of the study
#

def write_overlapping_characteristics(data_dir, results_dir):
    """ Writes characteristics of the overlapping datasets toa file, takes in directory of the X and y
    matrices, will place csv file output there too"""

    def write_combo_characteristic(cols, description):
        """
        Inner function to write a characteristic based on multiple binary columns
        :param cols:  columns to combine
        :param description: definition to print to table
        :return: calls write_characteristic_perc with needed info
        """
        combo_col_sd = X_sd_extval[cols].max(axis=1)
        combo_col_cd = X_cb_extval[cols].max(axis=1)

        sd_val = combo_col_sd.value_counts().to_dict()[1]
        cb_val = combo_col_cd.value_counts().to_dict()[1]
        write_characteristic_perc(description, sd_val, sd_n, cb_val, cb_n, f)

    # Read Matrices as pandas dataframes
    X_cb_extval = pd.read_csv(os.path.join(data_dir, "X_test_cb_extval.csv"))
    X_sd_extval = pd.read_csv(os.path.join(data_dir, "X_77_qlesq_sr__final_extval.csv"))
    y_cb_extval = pd.read_csv(os.path.join(data_dir, "canbind_qlesq_y__targets.csv"))
    y_sd_extval = pd.read_csv(os.path.join(data_dir, "y_qlesq_77__final__targets.csv"))

    # Read the pre-overlapping matrices for some data needed for certain stuff. Manually point to these older
    # pre-generate overlapping files
    cb_pre = pd.read_csv(os.path.join("../data/canbind_data/" + "canbind_imputed.csv"))
    sd_pre = pd.read_csv(os.path.join(data_dir, "X_77_qlesq_sr__final.csv"))

    # Filter out a few CAN-BIND subjects who weren't included in our study/didn't get y-gen
    X_cb_extval = X_cb_extval[X_cb_extval.index.isin(list(X_cb_extval.index.difference(y_cb_extval.index))) == False]
    cb_pre = cb_pre[cb_pre.index.isin(list(cb_pre.index.difference(y_cb_extval.index))) == False]

    X_cb_extval['Employed'] = cb_pre['EMPLOY_STATUS_1.0']
    X_sd_extval['Employed'] = sd_pre[['dm01_enroll__empl||3.0', 'dm01_enroll__empl||4.0']].max(axis=1)

    # Ensure number of subjects same between X and y matrices
    assert len(X_cb_extval) == len(y_cb_extval) and \
           len(X_sd_extval) == len(y_sd_extval) and \
           len(sd_pre) == len(y_sd_extval) and \
           len(cb_pre) == len(y_cb_extval), "X and y require same length"

    # Store n in our STAR*D and CAN-BIN datasets
    sd_n = len(X_sd_extval)
    cb_n = len(X_cb_extval)

    # Open file to write to. Write manually to allow strings
    f = open(results_dir + "/" + "paper_overlapping_characteristics.csv", "w")
    f.write("Characteristic, STAR*D,, CAN-BIND,\n")

    # Wrtie line for n and %
    f.write(',n,%,n,%\n')

    # Female:Male. Write manually as aytypical
    defin = "Female:Male"

    female_sd = X_sd_extval['SEX_female:::gender||F'].value_counts().to_dict()[1]
    male_sd = X_sd_extval['SEX_male:::gender||M'].value_counts().to_dict()[1]
    female_cb = X_cb_extval['SEX_female:::gender||F'].value_counts().to_dict()[1]
    male_cb = X_cb_extval['SEX_male:::gender||M'].value_counts().to_dict()[1]

    female_sd_perc = "{:.1f}".format(100 * female_sd / sd_n)
    male_sd_perc = "{:.1f}".format(100 * male_sd / sd_n)
    female_cb_perc = "{:.1f}".format(100 * female_cb / cb_n)
    male_cb_perc = "{:.1f}".format(100 * male_cb / cb_n)

    f.write(defin + "," + str(female_sd) + ":" + str(male_sd) + " ," + female_sd_perc + ":" + male_sd_perc + ",")
    f.write(str(female_cb) + ":" + str(male_cb) + " ," + female_cb_perc + ":" + male_cb_perc + "\n")

    # Married/Domestic Partnership
    partnered_cols = ['MRTL_STATUS_Married:::dm01_enroll__marital||3.0',
                      'MRTL_STATUS_Domestic Partnership:::dm01_enroll__marital||2.0']

    write_combo_characteristic(partnered_cols, "Married/Domestic Partnership")

    # Never Married/Divorced/Seperated/Widowed
    single_cols = ['MRTL_STATUS_Widowed:::dm01_enroll__marital||6.0',
                   'MRTL_STATUS_Divorced:::dm01_enroll__marital||5.0',
                   'MRTL_STATUS_Separated:::dm01_enroll__marital||4.0',
                   'MRTL_STATUS_Never Married:::dm01_enroll__marital||1.0']

    write_combo_characteristic(single_cols, "Never Married/Divorced/Seperated/Widowed")

    # Working/Student
    sd = X_sd_extval['EMPLOY_STATUS_1.0:::dm01_enroll__empl||3.0'].value_counts().to_dict()[1]
    cb = X_cb_extval['EMPLOY_STATUS_1.0:::dm01_enroll__empl||3.0'].value_counts().to_dict()[1]
    write_characteristic_perc("Working/Student", sd, sd_n, cb, cb_n, f)

    # Unemployed/Disabled/Retired
    not_work_cols = ['EMPLOY_STATUS_5.0:::dm01_enroll__empl||2.0',
                     'EMPLOY_STATUS_2.0:::dm01_enroll__empl||1.0',
                     'EMPLOY_STATUS_7.0:::dm01_enroll__empl||6.0']
    write_combo_characteristic(single_cols, "Unemployed/Disabled/Retired")

    # Any substance-use
    substance_cols = ['MINI_ALCHL_ABUSE_TIME:::phx01__alcoh||1.0',
                      'MINI_SBSTNC_ABUSE_NONALCHL_TIME:::phx01__amphet||1.0']

    write_combo_characteristic(substance_cols, "Any Substance Use Disorder")

    # Any anxiety disorder
    col = ':::imput_anyanxiety'
    defin = "Any Anxiety Disorder"
    sd = X_sd_extval[col].value_counts().to_dict()[1]
    cb = X_cb_extval[col].value_counts().to_dict()[1]
    write_characteristic_perc(defin, sd, sd_n, cb, cb_n, f)

    # line for mean and %
    f.write(', Mean, SD, Mean, SD\n')

    # Education
    defin = "Years of education"
    edu_years_mn_sd = X_sd_extval['EDUC:::dm01_enroll__educat'].mean()
    edu_years_mn_cb = X_cb_extval['EDUC:::dm01_enroll__educat'].mean()

    edu_years_std_sd = X_sd_extval['EDUC:::dm01_enroll__educat'].std()
    edu_years_std_cb = X_cb_extval['EDUC:::dm01_enroll__educat'].std()
    write_characteristic_cust(defin, edu_years_mn_sd, edu_years_std_sd, edu_years_mn_cb, edu_years_std_cb, f)

    # Employment Status: Hours worked if employed
    hrs_wrk_mn_sd = X_sd_extval[X_sd_extval['Employed'] > 0]['LAM_2_baseline:::wpai01__wpai_totalhrs'].mean()
    hrs_wrk_mn_cb = X_cb_extval[X_cb_extval['Employed'] > 0]['LAM_2_baseline:::wpai01__wpai_totalhrs'].mean()

    hrs_wrk_std_sd = X_sd_extval[X_sd_extval['Employed'] > 0]['LAM_2_baseline:::wpai01__wpai_totalhrs'].std()
    hrs_wrk_std_cb = X_cb_extval[X_cb_extval['Employed'] > 0]['LAM_2_baseline:::wpai01__wpai_totalhrs'].std()
    defin = "Hours worked over last two weeks if employed"
    f.write(defin + "," + "{:.1f}".format(hrs_wrk_mn_sd) + " ," + "{:.1f}".format(hrs_wrk_std_sd) + ",")
    f.write("{:.1f}".format(hrs_wrk_mn_cb) + " ," + "{:.1f}".format(hrs_wrk_std_cb) + ",\n")

    # Employment Status: Hours missed if employed
    hrs_msd_mn_sd = X_sd_extval[X_sd_extval['Employed'] > 0]['LAM_3_baseline:::wpai01__wpai02'].mean()
    hrs_msd_mn_cb = X_cb_extval[X_cb_extval['Employed'] > 0]['LAM_3_baseline:::wpai01__wpai02'].mean()

    hrs_msd_std_sd = X_sd_extval[X_sd_extval['Employed'] > 0]['LAM_3_baseline:::wpai01__wpai02'].std()
    hrs_msd_std_cb = X_cb_extval[X_cb_extval['Employed'] > 0]['LAM_3_baseline:::wpai01__wpai02'].std()
    defin = "Hours missed from illness over last two weeks if employed"
    f.write(defin + "," + "{:.1f}".format(hrs_msd_mn_sd) + "," + "{:.1f}".format(hrs_msd_std_sd) + ",")
    f.write("{:.1f}".format(hrs_msd_mn_cb) + " ," + "{:.1f}".format(hrs_msd_std_cb) + "\n")

    # Have some extra depression characteristics used in last paper, will skip for now
    include_depression_detailed = False
    if include_depression_detailed:
        # Age at onset of first depression PSYHIS_MDD_AGE:::phx01__dage
        defin = "Age in years at onset of depression (SD)"
        col = 'PSYHIS_MDD_AGE:::phx01__dage'
        sd = X_sd_extval[col].mean()
        cb = X_cb_extval[col].mean()

        sd_br = X_sd_extval[col].std()
        cb_br = X_cb_extval[col].std()
        write_characteristic_cust(defin, sd, sd_br, cb, cb_br, f)

        # Prior depressive episode
        col = 'PSYHIS_MDD_PREV:::'
        defin = "Prior depressive episode (%)"
        sd = X_sd_extval[col].value_counts().to_dict()[1]
        cb = X_cb_extval[col].value_counts().to_dict()[1]
        write_characteristic_perc(defin, sd, sd_n, cb, cb_n, f)

        # No. prior episodes. Subtract 1 as the col counts current episode
        defin = "No. of prior depressive episodes (SD)"
        col = 'PSYHIS_MDE_NUM:::phx01__epino'
        sd = X_sd_extval[col].sub(1).mean()
        cb = X_cb_extval[col].sub(1).mean()

        sd_br = X_sd_extval[col].sub(1).std()
        cb_br = X_cb_extval[col].sub(1).std()
        write_characteristic_cust(defin, sd, sd_br, cb, cb_br, f)

        # Current episode duration. Might want to cat this like in the paper
        defin = "Current episode duration in months (SD)"
        col = 'PSYHIS_MDE_EP_DUR_MO:::phx01__episode_date'
        sd = X_sd_extval[col].mean()
        cb = X_cb_extval[col].mean()

        sd_br = X_sd_extval[col].std()
        cb_br = X_cb_extval[col].std()
        write_characteristic_cust(defin, sd, sd_br, cb, cb_br, f)

    # QIDS-SR Total Baseline
    col = 'QIDS_OVERL_SEVTY_baseline:::qids01_w0sr__qstot'
    defin = "Baseline QIDS-SR Total Score"
    sd = X_sd_extval[col].mean()
    cb = X_cb_extval[col].mean()

    sd_br = X_sd_extval[col].std()
    cb_br = X_cb_extval[col].std()
    write_characteristic_cust(defin, sd, sd_br, cb, cb_br, f)

    # QIDS-SR Total Week 2
    col = 'QIDS_OVERL_SEVTY_week2:::qids01_w2sr__qstot'
    defin = "Week 2 QIDS-SR Total Score"
    sd = X_sd_extval[col].mean()
    cb = X_cb_extval[col].mean()

    sd_br = X_sd_extval[col].std()
    cb_br = X_cb_extval[col].std()
    write_characteristic_cust(defin, sd, sd_br, cb, cb_br, f)

    # QLESQ-SF Total Baseline
    col = 'QLESQA_TOT_QLESQB_TOT_merged:::'
    defin = "Baseline QLESQ-SF Total Score"
    sd = X_sd_extval[col].mean()
    cb = X_cb_extval[col].mean()

    sd_br = X_sd_extval[col].std()
    cb_br = X_cb_extval[col].std()
    write_characteristic_cust(defin, sd, sd_br, cb, cb_br, f)

    f.close()


def write_characteristic_perc(defin, sd, sd_n, cb, cb_n, f):
    """ Helper function that writes a line of the characteristics file
    defin -- string definition of the characteristic
    sd --  integer number of a certain characteristic
    sdtot -- int of total number of subjects in the STAR*D cohort
    cb -- integer number of a certain characteristic
    cbtot -- int of total number of subjects in the CANBIND cohort
    f -- file to write to"""

    # Calculate percentages
    sdperc = "{:.1f}".format(100 * sd / sd_n)
    cbperc = "{:.1f}".format(100 * cb / cb_n)

    # Write to file
    f.write(defin + "," + str(sd) + " ," + sdperc + ", " + str(cb) + " ," + cbperc + "\n")


def write_characteristic_cust(defin, sd, sd_br, cb, cb_br, f):
    """ Helper function that writes a line of the characteristics file, allowing custom values to be entered into the parens
    defin -- string definition of the characteristic
    sd -- STAR*D value to be outside parens when written
    sd_br -- STAR*D value written inside the parens
    cb -- CANBIND value outside the parens
    cb_br -- CANBIND value inside the parens
    f -- file to write to"""

    # Write to file
    f.write(defin + "," + "{:.1f}".format(sd) + " ," + "{:.1f}".format(sd_br) + ", " + "{:.1f}".format(
        cb) + " ," + "{:.1f}".format(cb_br) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        data_dir = sys.argv[1]

    if len(sys.argv) == 1:
        data_dir = "../data/modelling/"
        results_dir = "../results/table_1_cohort_char/"
        write_overlapping_characteristics(data_dir, results_dir)
    else:
        print("Enter valid argument\n"
              "\t path: the path to a real directory containing the final datasets\n"
              "\t path: the path to a real directory to put the output files ")
