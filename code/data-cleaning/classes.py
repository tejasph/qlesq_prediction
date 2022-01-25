# classes.py
import pandas as pd 

# Used in selection of y values
class subject_selector():

    def __init__(self, df):
        self.df = df.copy()
#         self.group_df = df.groupby('subjectkey')
        
    def get_random_subject(self):
        
        random_id = list(self.df.subjectkey.sample(1))[0]
        return self.df[self.df['subjectkey'] == random_id].sort_values(by = 'days_baseline', ascending = True)
    
    def get_specific_subject(self, chosen_id):
        
        return self.df[self.df['subjectkey'] == chosen_id].sort_values(by = 'days_baseline', ascending = True)
    
    def filter_time_window(self, min_cutoff = -1, max_cutoff = 77): # for some reason, there is 1 days baseline entry of -1
        self.df = self.df[(self.df['days_baseline']>= min_cutoff) & (self.df['days_baseline'] <= max_cutoff)]
        print(self.df.shape)
        
    def filter_NA(self, subset_col = ['totqlesq', 'level']):
        self.df = self.df.dropna(subset = subset_col)
        print(self.df.shape)
    
    def filter_duplicates(self):
        self.df = self.df.drop_duplicates()
        print(self.df.shape)
    
    def filter_lvl(self, lvl):    
        self.df = self.df.drop(self.df[self.df['level'] == lvl].index)
        print(self.df.shape)
        
    def filter_inappropriate_level_2(self):
        
        bad_ids = []
        grouped_df = self.df.groupby('subjectkey')
        
        for id, data in grouped_df:
            for row, col in data.iterrows():
    
                if data[(data['days_baseline'] < 21) & (data['level'] == 'Level 2')].shape[0] > 0:
                    bad_ids.append(id)
                   
        self.df = self.df[~self.df['subjectkey'].isin(bad_ids)]
        print(self.df.shape)
        
    def get_relevant_ids(self):
        
        group = self.df.groupby('subjectkey')
        relevant_ids = []
        for id, data in group:
#             data = data[data['days_baseline'] <= 77] # only level 1 should expect ~3000 # 78-91

            sorted_data = data.sort_values(by = ['days_baseline'], ascending = True)

            if data.shape[0] <= 1:
                continue

            baseline = sorted_data.iloc[0]['totqlesq']
            start_day = sorted_data.iloc[0]['days_baseline']
            end_score = sorted_data.iloc[-1]['totqlesq']
            end_day = sorted_data.iloc[-1]['days_baseline']
            end_lvl = sorted_data.iloc[-1]['level']

            if start_day >= 21:  #8-21
                continue

            if end_day <= 21 or end_day >=77:
                continue

            relevant_ids.append(id)
        print(f"Number of ids that fit criteria: {len(relevant_ids)}")
        return relevant_ids