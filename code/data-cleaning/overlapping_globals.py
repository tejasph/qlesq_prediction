"""
This is a globals file storing the configuration for managing the generation of the overlapping matrices

NOTES: ***********************
- All names of columns are standardized to uppercase.
- 
"""
NEW_FEATURES_STARD = ['PSYHIS_MDD_PREV:::',
    'imput_QIDS_SR_sleep_domain_week0:::',
    'imput_QIDS_SR_sleep_domain_week2:::',
    'imput_QIDS_SR_appetite_domain_week0:::',
    'imput_QIDS_SR_appetite_domain_week2:::',
    'imput_QIDS_SR_psychomot_domain_week0:::',
    'imput_QIDS_SR_psychomot_domain_week2:::',
    'imput_QIDS_SR_overeating_week0:::',
    'imput_QIDS_SR_overeating_week2:::',
    'imput_QIDS_SR_insomnia_week0:::',
    'imput_QIDS_SR_insomnia_week2:::',
    'imput_QIDS_SR_perc_change:::'
    'QIDS_ATYPICAL_baseline:::'
    'QIDS_ATYPICAL_week2:::'
    'QLESQA_TOT_QLESQB_TOT_merged:::' # Has a qlesq tot but that one uses items 14 and 15
    ]

NEW_FEATURES_CANBIND = [':::imput_anyanxiety',
    'imput_QIDS_SR_sleep_domain_week0:::',
    'imput_QIDS_SR_sleep_domain_week2:::',
    'imput_QIDS_SR_appetite_domain_week0:::',
    'imput_QIDS_SR_appetite_domain_week2:::',
    'imput_QIDS_SR_psychomot_domain_week0:::',
    'imput_QIDS_SR_psychomot_domain_week2:::',
    'imput_QIDS_SR_overeating_week0:::',
    'imput_QIDS_SR_overeating_week2:::',
    'imput_QIDS_SR_insomnia_week0:::',
    'imput_QIDS_SR_insomnia_week2:::',
    'imput_QIDS_SR_perc_change:::'
    ]

QIDS_STARD_TO_CANBIND_DICT = {
   'qids01_w0sr__vsoin':'QIDS_SR_1_baseline',
	'qids01_w0sr__vmnin':'QIDS_SR_2_baseline',
	'qids01_w0sr__vemin':'QIDS_SR_3_baseline',
	'qids01_w0sr__vhysm':'QIDS_SR_4_baseline',
	'qids01_w0sr__vmdsd':'QIDS_SR_5_baseline',
	'qids01_w0sr__vapdc':'QIDS_SR_6_baseline',
	'qids01_w0sr__vapin':'QIDS_SR_7_baseline',
	'qids01_w0sr__vwtdc':'QIDS_SR_8_baseline',
	'qids01_w0sr__vwtin':'QIDS_SR_9_baseline',
	'qids01_w0sr__vcntr':'QIDS_SR_10_baseline',
	'qids01_w0sr__vvwsf':'QIDS_SR_11_baseline',
	'qids01_w0sr__vsuic':'QIDS_SR_12_baseline',
	'qids01_w0sr__vintr':'QIDS_SR_13_baseline',
	'qids01_w0sr__vengy':'QIDS_SR_14_baseline',
	'qids01_w0sr__vslow':'QIDS_SR_15_baseline',
	'qids01_w0sr__vagit':'QIDS_SR_16_baseline',
	'qids01_w2sr__vsoin':'QIDS_SR_1_week2',
	'qids01_w2sr__vmnin':'QIDS_SR_2_week2',
	'qids01_w2sr__vemin':'QIDS_SR_3_week2',
	'qids01_w2sr__vhysm':'QIDS_SR_4_week2',
	'qids01_w2sr__vmdsd':'QIDS_SR_5_week2',
	'qids01_w2sr__vapdc':'QIDS_SR_6_week2',
	'qids01_w2sr__vapin':'QIDS_SR_7_week2',
	'qids01_w2sr__vwtdc':'QIDS_SR_8_week2',
	'qids01_w2sr__vwtin':'QIDS_SR_9_week2',
	'qids01_w2sr__vcntr':'QIDS_SR_10_week2',
	'qids01_w2sr__vvwsf':'QIDS_SR_11_week2',
	'qids01_w2sr__vsuic':'QIDS_SR_12_week2',
	'qids01_w2sr__vintr':'QIDS_SR_13_week2',
	'qids01_w2sr__vengy':'QIDS_SR_14_week2',
	'qids01_w2sr__vslow':'QIDS_SR_15_week2',
	'qids01_w2sr__vagit':'QIDS_SR_16_week2',
   'qids01_w0sr__qstot':'QIDS_OVERL_SEVTY_baseline',
   'qids01_w2sr__qstot':'QIDS_OVERL_SEVTY_week2'}


                        
HEADER_CONVERSION_DICT = {
        'SUBJLABEL':'SUBJLABEL:::subjectkey',
        'EDUC':'EDUC:::dm01_enroll__educat',
        'HSHLD_INCOME':'HSHLD_INCOME:::dm01_w0__totincom',
        'MINI_AGRPHOBIA_TIME':'MINI_AGRPHOBIA_TIME:::phx01__pd_ag',
        'MINI_ALCHL_ABUSE_TIME':'MINI_ALCHL_ABUSE_TIME:::phx01__alcoh||1.0',
        'MINI_APD_TIME':'MINI_APD_TIME:::phx01__pd_antis',
        'MINI_GAD_TIME':'MINI_GAD_TIME:::phx01__gad_phx',
        'MINI_AN_BINGE_TIME':'MINI_AN_BINGE_TIME:::phx01__anorexia',
        'MINI_BN_TIME':'MINI_BN_TIME:::phx01__bulimia||2/5',
        'MINI_OCD_TIME':'MINI_OCD_TIME:::phx01__ocd_phx',
        'MINI_PD_DX':'MINI_PD_DX:::phx01__pd_noag',
        'MINI_PTSD_TIME':'MINI_PTSD_TIME:::phx01__psd',
        'MINI_SBSTNC_ABUSE_NONALCHL_TIME':'MINI_SBSTNC_ABUSE_NONALCHL_TIME:::phx01__amphet||1.0',
        'MINI_SOCL_PHOBIA_DX':'MINI_SOCL_PHOBIA_DX:::phx01__soc_phob',
        'PSYHIS_FH':'PSYHIS_FH:::phx01__dep',
        'PSYHIS_MDD_AGE':'PSYHIS_MDD_AGE:::phx01__dage',
        'PSYHIS_MDD_PREV':'PSYHIS_MDD_PREV:::',
        'PSYHIS_MDE_EP_DUR_MO':'PSYHIS_MDE_EP_DUR_MO:::phx01__episode_date',
        'PSYHIS_MDE_NUM':'PSYHIS_MDE_NUM:::phx01__epino',
        'QLESQ_1A_1_baseline_QLESQ_1B_1_baseline_merged':'QLESQ_1A_1_baseline_QLESQ_1B_1_baseline_merged:::qlesq01__qlesq01',
        'QLESQ_1A_2_baseline_QLESQ_1B_2_baseline_merged':'QLESQ_1A_2_baseline_QLESQ_1B_2_baseline_merged:::qlesq01__qlesq02',
        'QLESQ_1A_3_baseline_QLESQ_1B_3_baseline_merged':'QLESQ_1A_3_baseline_QLESQ_1B_3_baseline_merged:::qlesq01__qlesq03',
        'QLESQ_1A_4_baseline_QLESQ_1B_4_baseline_merged':'QLESQ_1A_4_baseline_QLESQ_1B_4_baseline_merged:::qlesq01__qlesq04',
        'QLESQ_1A_5_baseline_QLESQ_1B_5_baseline_merged':'QLESQ_1A_5_baseline_QLESQ_1B_5_baseline_merged:::qlesq01__qlesq05',
        'QLESQ_1A_6_baseline_QLESQ_1B_6_baseline_merged':'QLESQ_1A_6_baseline_QLESQ_1B_6_baseline_merged:::qlesq01__qlesq06',
        'QLESQ_1A_7_baseline_QLESQ_1B_7_baseline_merged':'QLESQ_1A_7_baseline_QLESQ_1B_7_baseline_merged:::qlesq01__qlesq07',
        'QLESQ_1A_8_baseline_QLESQ_1B_8_baseline_merged':'QLESQ_1A_8_baseline_QLESQ_1B_8_baseline_merged:::qlesq01__qlesq08',
        'QLESQ_1A_9_baseline_QLESQ_1B_9_baseline_merged':'QLESQ_1A_9_baseline_QLESQ_1B_9_baseline_merged:::qlesq01__qlesq09',
        'QLESQ_1A_10_baseline_QLESQ_1B_10_baseline_merged':'QLESQ_1A_10_baseline_QLESQ_1B_10_baseline_merged:::qlesq01__qlesq10',
        'QLESQ_1A_11_baseline_QLESQ_1B_11_baseline_merged':'QLESQ_1A_11_baseline_QLESQ_1B_11_baseline_merged:::qlesq01__qlesq11',
        'QLESQ_1A_12_baseline_QLESQ_1B_12_baseline_merged':'QLESQ_1A_12_baseline_QLESQ_1B_12_baseline_merged:::qlesq01__qlesq12',
        'QLESQ_1A_13_baseline_QLESQ_1B_13_baseline_merged':'QLESQ_1A_13_baseline_QLESQ_1B_13_baseline_merged:::qlesq01__qlesq13',
        'QLESQ_1A_14_baseline_QLESQ_1B_14_baseline_merged':'QLESQ_1A_14_baseline_QLESQ_1B_14_baseline_merged:::qlesq01__qlesq14',
        'QLESQ_1A_16_baseline_QLESQ_1B_16_baseline_merged':'QLESQ_1A_16_baseline_QLESQ_1B_16_baseline_merged:::qlesq01__qlesq16',
        'QLESQA_TOT_baseline_QLESQB_TOT_baseline_merged':'QLESQA_TOT_QLESQB_TOT_merged:::',
        'SDS_1_1_baseline':'SDS_1_1_baseline:::wsas01__wsas01',
        'SDS_2_1_baseline':'SDS_2_1_baseline:::wsas01__wsas03',
        'SDS_3_1_baseline':'SDS_3_1_baseline:::wsas01__wsas02',
        'LAM_2_baseline':'LAM_2_baseline:::wpai01__wpai_totalhrs',
        'LAM_3_baseline':'LAM_3_baseline:::wpai01__wpai02',
        'SEX_female':'SEX_female:::gender||F',
        'SEX_male':'SEX_male:::gender||M',
        'AGE':'AGE:::interview_age',
        'MRTL_STATUS_Divorced':'MRTL_STATUS_Divorced:::dm01_enroll__marital||5.0',
        'MRTL_STATUS_Domestic Partnership':'MRTL_STATUS_Domestic Partnership:::dm01_enroll__marital||2.0',
        'MRTL_STATUS_Married':'MRTL_STATUS_Married:::dm01_enroll__marital||3.0',
        'MRTL_STATUS_Never Married':'MRTL_STATUS_Never Married:::dm01_enroll__marital||1.0',
        'MRTL_STATUS_Separated':'MRTL_STATUS_Separated:::dm01_enroll__marital||4.0',
        'MRTL_STATUS_Widowed':'MRTL_STATUS_Widowed:::dm01_enroll__marital||6.0',
        'EMPLOY_STATUS_1.0':'EMPLOY_STATUS_1.0:::dm01_enroll__empl||3.0',
        'EMPLOY_STATUS_2.0':'EMPLOY_STATUS_2.0:::dm01_enroll__empl||1.0',
        'EMPLOY_STATUS_5.0':'EMPLOY_STATUS_5.0:::dm01_enroll__empl||2.0',
        'EMPLOY_STATUS_7.0':'EMPLOY_STATUS_7.0:::dm01_enroll__empl||6.0',
        'QIDS_SR_1_baseline':'QIDS_SR_1_baseline:::qids01_w0sr__vsoin',
        'QIDS_SR_2_baseline':'QIDS_SR_2_baseline:::qids01_w0sr__vmnin',
        'QIDS_SR_3_baseline':'QIDS_SR_3_baseline:::qids01_w0sr__vemin',
        'QIDS_SR_4_baseline':'QIDS_SR_4_baseline:::qids01_w0sr__vhysm',
        'QIDS_SR_5_baseline':'QIDS_SR_5_baseline:::qids01_w0sr__vmdsd',
        'QIDS_SR_6_baseline':'QIDS_SR_6_baseline:::qids01_w0sr__vapdc',
        'QIDS_SR_7_baseline':'QIDS_SR_7_baseline:::qids01_w0sr__vapin',
        'QIDS_SR_8_baseline':'QIDS_SR_8_baseline:::qids01_w0sr__vwtdc',
        'QIDS_SR_9_baseline':'QIDS_SR_9_baseline:::qids01_w0sr__vwtin',
        'QIDS_SR_10_baseline':'QIDS_SR_10_baseline:::qids01_w0sr__vcntr',
        'QIDS_SR_11_baseline':'QIDS_SR_11_baseline:::qids01_w0sr__vvwsf',
        'QIDS_SR_12_baseline':'QIDS_SR_12_baseline:::qids01_w0sr__vsuic',
        'QIDS_SR_13_baseline':'QIDS_SR_13_baseline:::qids01_w0sr__vintr',
        'QIDS_SR_14_baseline':'QIDS_SR_14_baseline:::qids01_w0sr__vengy',
        'QIDS_SR_15_baseline':'QIDS_SR_15_baseline:::qids01_w0sr__vslow',
        'QIDS_SR_16_baseline':'QIDS_SR_16_baseline:::qids01_w0sr__vagit',
        'QIDS_OVERL_SEVTY_baseline':'QIDS_OVERL_SEVTY_baseline:::qids01_w0sr__qstot',
        'QIDS_SR_1_week2':'QIDS_SR_1_week2:::qids01_w2sr__vsoin',
        'QIDS_SR_2_week2':'QIDS_SR_2_week2:::qids01_w2sr__vmnin',
        'QIDS_SR_3_week2':'QIDS_SR_3_week2:::qids01_w2sr__vemin',
        'QIDS_SR_4_week2':'QIDS_SR_4_week2:::qids01_w2sr__vhysm',
        'QIDS_SR_5_week2':'QIDS_SR_5_week2:::qids01_w2sr__vmdsd',
        'QIDS_SR_6_week2':'QIDS_SR_6_week2:::qids01_w2sr__vapdc',
        'QIDS_SR_7_week2':'QIDS_SR_7_week2:::qids01_w2sr__vapin',
        'QIDS_SR_8_week2':'QIDS_SR_8_week2:::qids01_w2sr__vwtdc',
        'QIDS_SR_9_week2':'QIDS_SR_9_week2:::qids01_w2sr__vwtin',
        'QIDS_SR_10_week2':'QIDS_SR_10_week2:::qids01_w2sr__vcntr',
        'QIDS_SR_11_week2':'QIDS_SR_11_week2:::qids01_w2sr__vvwsf',
        'QIDS_SR_12_week2':'QIDS_SR_12_week2:::qids01_w2sr__vsuic',
        'QIDS_SR_13_week2':'QIDS_SR_13_week2:::qids01_w2sr__vintr',
        'QIDS_SR_14_week2':'QIDS_SR_14_week2:::qids01_w2sr__vengy',
        'QIDS_SR_15_week2':'QIDS_SR_15_week2:::qids01_w2sr__vslow',
        'QIDS_SR_16_week2':'QIDS_SR_16_week2:::qids01_w2sr__vagit',
        'QIDS_OVERL_SEVTY_week2':'QIDS_OVERL_SEVTY_week2:::qids01_w2sr__qstot',
        'subjectkey':'SUBJLABEL:::subjectkey',
        'dm01_enroll__educat':'EDUC:::dm01_enroll__educat',
        'dm01_w0__totincom':'HSHLD_INCOME:::dm01_w0__totincom',
        'phx01__pd_ag':'MINI_AGRPHOBIA_TIME:::phx01__pd_ag',
        'phx01__alcoh||1.0':'MINI_ALCHL_ABUSE_TIME:::phx01__alcoh||1.0',
        'phx01__pd_antis':'MINI_APD_TIME:::phx01__pd_antis',
        'phx01__gad_phx':'MINI_GAD_TIME:::phx01__gad_phx',
        'phx01__anorexia':'MINI_AN_BINGE_TIME:::phx01__anorexia',
        'phx01__bulimia||2/5':'MINI_BN_TIME:::phx01__bulimia||2/5',
        'phx01__ocd_phx':'MINI_OCD_TIME:::phx01__ocd_phx',
        'phx01__pd_noag':'MINI_PD_DX:::phx01__pd_noag',
        'phx01__psd':'MINI_PTSD_TIME:::phx01__psd',
        'phx01__amphet||1.0':'MINI_SBSTNC_ABUSE_NONALCHL_TIME:::phx01__amphet||1.0',
        'phx01__soc_phob':'MINI_SOCL_PHOBIA_DX:::phx01__soc_phob',
        'phx01__dep':'PSYHIS_FH:::phx01__dep',
        'phx01__dage':'PSYHIS_MDD_AGE:::phx01__dage',
        'phx01__episode_date':'PSYHIS_MDE_EP_DUR_MO:::phx01__episode_date',
        'phx01__epino':'PSYHIS_MDE_NUM:::phx01__epino',
        'qlesq01__qlesq01':'QLESQ_1A_1_baseline_QLESQ_1B_1_baseline_merged:::qlesq01__qlesq01',
        'qlesq01__qlesq02':'QLESQ_1A_2_baseline_QLESQ_1B_2_baseline_merged:::qlesq01__qlesq02',
        'qlesq01__qlesq03':'QLESQ_1A_3_baseline_QLESQ_1B_3_baseline_merged:::qlesq01__qlesq03',
        'qlesq01__qlesq04':'QLESQ_1A_4_baseline_QLESQ_1B_4_baseline_merged:::qlesq01__qlesq04',
        'qlesq01__qlesq05':'QLESQ_1A_5_baseline_QLESQ_1B_5_baseline_merged:::qlesq01__qlesq05',
        'qlesq01__qlesq06':'QLESQ_1A_6_baseline_QLESQ_1B_6_baseline_merged:::qlesq01__qlesq06',
        'qlesq01__qlesq07':'QLESQ_1A_7_baseline_QLESQ_1B_7_baseline_merged:::qlesq01__qlesq07',
        'qlesq01__qlesq08':'QLESQ_1A_8_baseline_QLESQ_1B_8_baseline_merged:::qlesq01__qlesq08',
        'qlesq01__qlesq09':'QLESQ_1A_9_baseline_QLESQ_1B_9_baseline_merged:::qlesq01__qlesq09',
        'qlesq01__qlesq10':'QLESQ_1A_10_baseline_QLESQ_1B_10_baseline_merged:::qlesq01__qlesq10',
        'qlesq01__qlesq11':'QLESQ_1A_11_baseline_QLESQ_1B_11_baseline_merged:::qlesq01__qlesq11',
        'qlesq01__qlesq12':'QLESQ_1A_12_baseline_QLESQ_1B_12_baseline_merged:::qlesq01__qlesq12',
        'qlesq01__qlesq13':'QLESQ_1A_13_baseline_QLESQ_1B_13_baseline_merged:::qlesq01__qlesq13',
        'qlesq01__qlesq14':'QLESQ_1A_14_baseline_QLESQ_1B_14_baseline_merged:::qlesq01__qlesq14',
        'qlesq01__qlesq16':'QLESQ_1A_16_baseline_QLESQ_1B_16_baseline_merged:::qlesq01__qlesq16',
        'wsas01__wsas01':'SDS_1_1_baseline:::wsas01__wsas01',
        'wsas01__wsas03':'SDS_2_1_baseline:::wsas01__wsas03',
        'wsas01__wsas02':'SDS_3_1_baseline:::wsas01__wsas02',
        'wpai01__wpai_totalhrs':'LAM_2_baseline:::wpai01__wpai_totalhrs',
        'wpai01__wpai02':'LAM_3_baseline:::wpai01__wpai02',
        'gender||F':'SEX_female:::gender||F',
        'gender||M':'SEX_male:::gender||M',
        'interview_age':'AGE:::interview_age',
        'dm01_enroll__marital||5.0':'MRTL_STATUS_Divorced:::dm01_enroll__marital||5.0',
        'dm01_enroll__marital||2.0':'MRTL_STATUS_Domestic Partnership:::dm01_enroll__marital||2.0',
        'dm01_enroll__marital||3.0':'MRTL_STATUS_Married:::dm01_enroll__marital||3.0',
        'dm01_enroll__marital||1.0':'MRTL_STATUS_Never Married:::dm01_enroll__marital||1.0',
        'dm01_enroll__marital||4.0':'MRTL_STATUS_Separated:::dm01_enroll__marital||4.0',
        'dm01_enroll__marital||6.0':'MRTL_STATUS_Widowed:::dm01_enroll__marital||6.0',
        'dm01_enroll__empl||3.0':'EMPLOY_STATUS_1.0:::dm01_enroll__empl||3.0',
        'dm01_enroll__empl||1.0':'EMPLOY_STATUS_2.0:::dm01_enroll__empl||1.0',
        'dm01_enroll__empl||2.0':'EMPLOY_STATUS_5.0:::dm01_enroll__empl||2.0',
        'dm01_enroll__empl||6.0':'EMPLOY_STATUS_7.0:::dm01_enroll__empl||6.0',
        'qids01_w0sr__vsoin':'QIDS_SR_1_baseline:::qids01_w0sr__vsoin',
        'qids01_w0sr__vmnin':'QIDS_SR_2_baseline:::qids01_w0sr__vmnin',
        'qids01_w0sr__vemin':'QIDS_SR_3_baseline:::qids01_w0sr__vemin',
        'qids01_w0sr__vhysm':'QIDS_SR_4_baseline:::qids01_w0sr__vhysm',
        'qids01_w0sr__vmdsd':'QIDS_SR_5_baseline:::qids01_w0sr__vmdsd',
        'qids01_w0sr__vapdc':'QIDS_SR_6_baseline:::qids01_w0sr__vapdc',
        'qids01_w0sr__vapin':'QIDS_SR_7_baseline:::qids01_w0sr__vapin',
        'qids01_w0sr__vwtdc':'QIDS_SR_8_baseline:::qids01_w0sr__vwtdc',
        'qids01_w0sr__vwtin':'QIDS_SR_9_baseline:::qids01_w0sr__vwtin',
        'qids01_w0sr__vcntr':'QIDS_SR_10_baseline:::qids01_w0sr__vcntr',
        'qids01_w0sr__vvwsf':'QIDS_SR_11_baseline:::qids01_w0sr__vvwsf',
        'qids01_w0sr__vsuic':'QIDS_SR_12_baseline:::qids01_w0sr__vsuic',
        'qids01_w0sr__vintr':'QIDS_SR_13_baseline:::qids01_w0sr__vintr',
        'qids01_w0sr__vengy':'QIDS_SR_14_baseline:::qids01_w0sr__vengy',
        'qids01_w0sr__vslow':'QIDS_SR_15_baseline:::qids01_w0sr__vslow',
        'qids01_w0sr__vagit':'QIDS_SR_16_baseline:::qids01_w0sr__vagit',
        'qids01_w0sr__qstot':'QIDS_OVERL_SEVTY_baseline:::qids01_w0sr__qstot',
        'qids01_w2sr__vsoin':'QIDS_SR_1_week2:::qids01_w2sr__vsoin',
        'qids01_w2sr__vmnin':'QIDS_SR_2_week2:::qids01_w2sr__vmnin',
        'qids01_w2sr__vemin':'QIDS_SR_3_week2:::qids01_w2sr__vemin',
        'qids01_w2sr__vhysm':'QIDS_SR_4_week2:::qids01_w2sr__vhysm',
        'qids01_w2sr__vmdsd':'QIDS_SR_5_week2:::qids01_w2sr__vmdsd',
        'qids01_w2sr__vapdc':'QIDS_SR_6_week2:::qids01_w2sr__vapdc',
        'qids01_w2sr__vapin':'QIDS_SR_7_week2:::qids01_w2sr__vapin',
        'qids01_w2sr__vwtdc':'QIDS_SR_8_week2:::qids01_w2sr__vwtdc',
        'qids01_w2sr__vwtin':'QIDS_SR_9_week2:::qids01_w2sr__vwtin',
        'qids01_w2sr__vcntr':'QIDS_SR_10_week2:::qids01_w2sr__vcntr',
        'qids01_w2sr__vvwsf':'QIDS_SR_11_week2:::qids01_w2sr__vvwsf',
        'qids01_w2sr__vsuic':'QIDS_SR_12_week2:::qids01_w2sr__vsuic',
        'qids01_w2sr__vintr':'QIDS_SR_13_week2:::qids01_w2sr__vintr',
        'qids01_w2sr__vengy':'QIDS_SR_14_week2:::qids01_w2sr__vengy',
        'qids01_w2sr__vslow':'QIDS_SR_15_week2:::qids01_w2sr__vslow',
        'qids01_w2sr__vagit':'QIDS_SR_16_week2:::qids01_w2sr__vagit',
        'qids01_w2sr__qstot':'QIDS_OVERL_SEVTY_week2:::qids01_w2sr__qstot',
        'imput_anyanxiety':':::imput_anyanxiety',
        'imput_idsc5w0':':::imput_idsc5w0',
        'imput_idsc5w2':':::imput_idsc5w2',
        'imput_idsc5pccg':':::imput_idsc5pccg',
        'imput_qidscpccg':':::imput_qidscpccg',
        'QIDS_ATYPICAL_baseline':'QIDS_ATYPICAL_baseline:::',
        'QIDS_ATYPICAL_week2':'QIDS_ATYPICAL_week2:::',
        }


CANBIND_OVERLAPPING_VALUE_CONVERSION_MAP = {
        
    "whitelist": [
      'SUBJLABEL',
		'EDUC',
		'HSHLD_INCOME',
		'MINI_AGRPHOBIA_TIME',
		'MINI_ALCHL_ABUSE_TIME',
		'MINI_APD_TIME',
		'MINI_GAD_TIME',
		'MINI_AN_BINGE_TIME',
		'MINI_BN_TIME',
		'MINI_OCD_TIME',
		'MINI_PD_DX',
		'MINI_PTSD_TIME',
		'MINI_SBSTNC_ABUSE_NONALCHL_TIME',
		'MINI_SOCL_PHOBIA_DX',
		'PSYHIS_FH',
		'PSYHIS_MDD_AGE',
		'PSYHIS_MDD_PREV',
		'PSYHIS_MDE_EP_DUR_MO',
		'PSYHIS_MDE_NUM',
		'QLESQ_1A_1_baseline_QLESQ_1B_1_baseline_merged',
		'QLESQ_1A_2_baseline_QLESQ_1B_2_baseline_merged',
		'QLESQ_1A_3_baseline_QLESQ_1B_3_baseline_merged',
		'QLESQ_1A_4_baseline_QLESQ_1B_4_baseline_merged',
		'QLESQ_1A_5_baseline_QLESQ_1B_5_baseline_merged',
		'QLESQ_1A_6_baseline_QLESQ_1B_6_baseline_merged',
		'QLESQ_1A_7_baseline_QLESQ_1B_7_baseline_merged',
		'QLESQ_1A_8_baseline_QLESQ_1B_8_baseline_merged',
		'QLESQ_1A_9_baseline_QLESQ_1B_9_baseline_merged',
		'QLESQ_1A_10_baseline_QLESQ_1B_10_baseline_merged',
		'QLESQ_1A_11_baseline_QLESQ_1B_11_baseline_merged',
		'QLESQ_1A_12_baseline_QLESQ_1B_12_baseline_merged',
		'QLESQ_1A_13_baseline_QLESQ_1B_13_baseline_merged',
		'QLESQ_1A_14_baseline_QLESQ_1B_14_baseline_merged',
		'QLESQ_1A_16_baseline_QLESQ_1B_16_baseline_merged',
		'QLESQA_TOT_baseline_QLESQB_TOT_baseline_merged',
		'SDS_1_1_baseline',
		'SDS_2_1_baseline',
		'SDS_3_1_baseline',
		'LAM_2_baseline',
		'LAM_3_baseline',
		'SEX_female',
		'SEX_male',
		'AGE',
		'MRTL_STATUS_Divorced',
		'MRTL_STATUS_Domestic Partnership',
		'MRTL_STATUS_Married',
		'MRTL_STATUS_Never Married',
		'MRTL_STATUS_Separated',
		'MRTL_STATUS_Widowed',
		'EMPLOY_STATUS_1.0',
		'EMPLOY_STATUS_2.0',
		'EMPLOY_STATUS_5.0',
		'EMPLOY_STATUS_7.0',
		'QIDS_SR_1_baseline',
		'QIDS_SR_2_baseline',
		'QIDS_SR_3_baseline',
		'QIDS_SR_4_baseline',
		'QIDS_SR_5_baseline',
		'QIDS_SR_6_baseline',
		'QIDS_SR_7_baseline',
		'QIDS_SR_8_baseline',
		'QIDS_SR_9_baseline',
		'QIDS_SR_10_baseline',
		'QIDS_SR_11_baseline',
		'QIDS_SR_12_baseline',
		'QIDS_SR_13_baseline',
		'QIDS_SR_14_baseline',
		'QIDS_SR_15_baseline',
		'QIDS_SR_16_baseline',
		'QIDS_OVERL_SEVTY_baseline',
		'QIDS_SR_1_week2',
		'QIDS_SR_2_week2',
		'QIDS_SR_3_week2',
		'QIDS_SR_4_week2',
		'QIDS_SR_5_week2',
		'QIDS_SR_6_week2',
		'QIDS_SR_7_week2',
		'QIDS_SR_8_week2',
		'QIDS_SR_9_week2',
		'QIDS_SR_10_week2',
		'QIDS_SR_11_week2',
		'QIDS_SR_12_week2',
		'QIDS_SR_13_week2',
		'QIDS_SR_14_week2',
		'QIDS_SR_15_week2',
		'QIDS_SR_16_week2',
		'QIDS_OVERL_SEVTY_week2',
		'QIDS_ATYPICAL_baseline',
		'QIDS_ATYPICAL_week2',
		'MINI_AN_TIME',
		'MINI_ALCHL_DPNDC_TIME',
		'MINI_SBSTNC_DPNDC_NONALCHL_TIME',
		'EMPLOY_STATUS_4.0',
		'EMPLOY_STATUS_6.0',
      'EMPLOY_STATUS_3.0'],    

    "multiply": {
        "description": "Multiply the value by the multiple specified.",
        "col_names": {
            #"PSYHIS_MDE_EP_DUR_MO": 30, Keep this as months so number smaller
        }
    },
    "other": {}
}
        
        
        
STARD_OVERLAPPING_VALUE_CONVERSION_MAP = {
# =============================================================================
#     "whitelist": ['epino', 'subjectkey', 'educat', 'pd_ag', 'pd_antis', 'gad_phx', 'anorexia', 'bulimia_0.0',
#                   'ocd_phx', 'pd_noag', 'psd', 'soc_phob', 'epino', 'qlesq01', 'qlesq02', 'qlesq03', 'qlesq04',
#                   'qlesq05', 'qlesq06', 'qlesq07', 'qlesq08', 'qlesq09', 'qlesq10', 'qlesq11', 'qlesq12', 'qlesq13',
#                   'qlesq14', 'qlesq16', 'totqlesq', 'interview_age', 'episode_date',
#                   'wsas01', 'wsas03', 'wsas02', 'wpai_totalhrs', 'wpai02', 'empl_2.0', 'empl_1.0', 'empl_3.0', 'episode_date',
#                   'dage', 'empl_1.0', 'empl_3.0', 'empl_5.0', 'empl_2.0','empl_4.0', 'empl_6.0', 'marital_5.0',
#                   'marital_2.0', 'marital_3.0', 'marital_1.0', 'marital_4.0', 'marital_6.0',
#                   'qstot_week0_Self Rating', 'qstot_week2_Self Rating'], # Left out: 'alcoh' (one hot encoded) 'totincom' (too sparse) 'empl_8.0' 'empl_14.0' (non existent in data)
# =============================================================================
    
    "whitelist" : ['subjectkey',
	'dm01_enroll__educat',
	'dm01_w0__totincom',
	'phx01__pd_ag',
	'phx01__alcoh||1.0',
	'phx01__pd_antis',
	'phx01__gad_phx',
	'phx01__anorexia',
	'phx01__bulimia||2/5',
	'phx01__ocd_phx',
	'phx01__pd_noag',
	'phx01__psd',
	'phx01__amphet||1.0',
	'phx01__soc_phob',
	'phx01__dep',
	'phx01__dage',
	'phx01__epino',
	'phx01__episode_date',
	'qlesq01__qlesq01',
	'qlesq01__qlesq02',
	'qlesq01__qlesq03',
	'qlesq01__qlesq04',
	'qlesq01__qlesq05',
	'qlesq01__qlesq06',
	'qlesq01__qlesq07',
	'qlesq01__qlesq08',
	'qlesq01__qlesq09',
	'qlesq01__qlesq10',
	'qlesq01__qlesq11',
	'qlesq01__qlesq12',
	'qlesq01__qlesq13',
	'qlesq01__qlesq14',
	'qlesq01__qlesq16',
	'wsas01__wsas01',
	'wsas01__wsas03',
	'wsas01__wsas02',
	'wpai01__wpai_totalhrs',
	'wpai01__wpai02',
	'gender||F',
	'gender||M',
	'interview_age',
	'dm01_enroll__marital||5.0',
	'dm01_enroll__marital||2.0',
	'dm01_enroll__marital||3.0',
	'dm01_enroll__marital||1.0',
	'dm01_enroll__marital||4.0',
	'dm01_enroll__marital||6.0',
	'dm01_enroll__empl||3.0',
	'dm01_enroll__empl||1.0',
	'dm01_enroll__empl||2.0',
	'dm01_enroll__empl||6.0',
	'qids01_w0sr__vsoin',
	'qids01_w0sr__vmnin',
	'qids01_w0sr__vemin',
	'qids01_w0sr__vhysm',
	'qids01_w0sr__vmdsd',
	'qids01_w0sr__vapdc',
	'qids01_w0sr__vapin',
	'qids01_w0sr__vwtdc',
	'qids01_w0sr__vwtin',
	'qids01_w0sr__vcntr',
	'qids01_w0sr__vvwsf',
	'qids01_w0sr__vsuic',
	'qids01_w0sr__vintr',
	'qids01_w0sr__vengy',
	'qids01_w0sr__vslow',
	'qids01_w0sr__vagit',
	'qids01_w0sr__qstot',
	'qids01_w2sr__vsoin',
	'qids01_w2sr__vmnin',
	'qids01_w2sr__vemin',
	'qids01_w2sr__vhysm',
	'qids01_w2sr__vmdsd',
	'qids01_w2sr__vapdc',
	'qids01_w2sr__vapin',
	'qids01_w2sr__vwtdc',
	'qids01_w2sr__vwtin',
	'qids01_w2sr__vcntr',
	'qids01_w2sr__vvwsf',
	'qids01_w2sr__vsuic',
	'qids01_w2sr__vintr',
	'qids01_w2sr__vengy',
	'qids01_w2sr__vslow',
	'qids01_w2sr__vagit',
	'qids01_w2sr__qstot',
	'phx01__alcoh||2.0',
	'phx01__bulimia||4',
	'phx01__bulimia||3',
	'phx01__amphet||2.0',
	'phx01__cannibis||1.0',
	'phx01__cannibis||2.0',
	'phx01__opioid||1.0',
	'phx01__opioid||2.0',
	'phx01__ax_cocaine||1.0',
	'phx01__ax_cocaine||2.0',
	'phx01__deppar',
	'phx01__depsib',
	'phx01__depchld',
	'phx01__bip',
	'phx01__bippar',
	'phx01__bipsib',
	'phx01__bipchld',
	'phx01__alcohol',
	'phx01__alcpar',
	'phx01__alcsib',
	'phx01__alcchld',
	'phx01__drug_phx',
	'phx01__drgpar',
	'phx01__drgsib',
	'phx01__drgchld',
	'phx01__suic_phx',
	'phx01__suicpar',
	'phx01__suicsib',
	'phx01__suicchld',
	'dm01_enroll__empl||4.0',
	'dm01_enroll__empl||5.0',
    'phx01__specphob'
            ],              
    "multiply": {
        "description": "Multiply the value by the multiple specified.",
        "col_names": {
            "wsas01__wsas01": 1.25,
            "wsas01__wsas03": 1.25,
            "wsas01__wsas02": 1.25,
            "wpai01__wpai_totalhrs": 2,
            "wpai01__wpai02": 2,
            "phx01__episode_date": 0.0333, # /30. Double check doesn't also need negative
            "interview_age": 0.082222 #/12 to convert month age to year
        }
    },
    "other": {}
    ##"blacklist": ['empl_5.0', 'empl_4.0'] # Use to remove after done processing
}
        
STARD_TO_DROP = ['phx01__alcoh||2.0',
	'phx01__bulimia||4',
	'phx01__bulimia||3',
	'phx01__amphet||2.0',
	'phx01__cannibis||1.0',
	'phx01__cannibis||2.0',
	'phx01__opioid||1.0',
	'phx01__opioid||2.0',
	'phx01__ax_cocaine||1.0',
	'phx01__ax_cocaine||2.0',
	'phx01__deppar',
	'phx01__depsib',
	'phx01__depchld',
	'phx01__bip',
	'phx01__bippar',
	'phx01__bipsib',
	'phx01__bipchld',
	'phx01__alcohol',
	'phx01__alcpar',
	'phx01__alcsib',
	'phx01__alcchld',
	'phx01__drug_phx',
	'phx01__drgpar',
	'phx01__drgsib',
	'phx01__drgchld',
	'phx01__suic_phx',
	'phx01__suicpar',
	'phx01__suicsib',
	'phx01__suicchld',
   'phx01__specphob',
	'dm01_enroll__empl||4.0',
	'dm01_enroll__empl||5.0'
    ]