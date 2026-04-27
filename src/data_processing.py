import pandas as pd

#DATA PREPROCESSING

class DataPreprocessor:
    def __init__(self, raw_df : pd.DataFrame):
        self.raw_df = raw_df
        self.selected_cols = ['sex', 'age', 'race', 'juv_fel_count',
        'juv_misd_count', 'juv_other_count', 'priors_count', 'c_jail_in', 
        'c_jail_out','c_charge_degree', 'is_recid', 'c_offense_date']


    def preprocess(self) -> (pd.DataFrame, pd.DataFrame):
        silver_df = self.raw2silver(self.raw_df)
        gold_df = self.silver2gold(silver_df)
        mlready_df = self.gold2ml(gold_df)
        return mlready_df, gold_df #send gold to compute metrics 
    
    def raw2silver(self, raw_df : pd.DataFrame) -> pd.DataFrame:
        silver_df = raw_df
        silver_df = silver_df[self.selected_cols]
        return silver_df

    def silver2gold(self, silver_df : pd.DataFrame) -> pd.DataFrame:
        gold_df = silver_df
        gold_df = self._create_gold_features(gold_df)
        gold_df = self._filter_data(gold_df)

        return gold_df
    
    def gold2ml(self, gold_df : pd.DataFrame) -> pd.DataFrame: 
        mlready_df = gold_df
        #One-hot encode categorical features
        mlready_df = pd.get_dummies(mlready_df, columns=["sex", "c_charge_degree", "race"])
        return mlready_df

    def _create_gold_features(self,gold_df : pd.DataFrame) -> pd.DataFrame : 
        gold_df['jail_duration'] = (pd.to_datetime(gold_df['c_jail_out']) - pd.to_datetime(gold_df['c_jail_in'])) / pd.Timedelta(days=1)
        gold_df['time_btw_offense_and_jail'] = (pd.to_datetime(gold_df['c_jail_in']) - pd.to_datetime(gold_df['c_offense_date']))/ pd.Timedelta(days=1)
        return gold_df

    def _filter_data(self, gold_df : pd.DataFrame) -> pd.DataFrame:
        gold_df = gold_df[gold_df['jail_duration'] >= 0]
        gold_df = gold_df[gold_df['time_btw_offense_and_jail'] >= 0]
        gold_df = gold_df.drop(columns=['c_jail_in', 'c_jail_out', 'c_offense_date'])
        return gold_df