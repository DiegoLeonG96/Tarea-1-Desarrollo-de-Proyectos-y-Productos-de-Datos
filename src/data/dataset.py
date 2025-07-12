import pandas as pd
from src.config import Configurations
from src.features.build_features import synthetic_features
import os 

class Dataset():
    def __init__(self):
        self.configs = Configurations()
    
    def change_configs(self, new_configs: dict):
        self.configs.change_configs(new_configs)

    def load_raw_data(self, month: str):
        taxi = pd.read_parquet(f"{self.configs.get_dict()['DATASET_BASE_URL']}_{month}.parquet")
        os.makedirs(self.configs.get_dict()['RAW_DATA_DIR'], exist_ok=True)
        taxi.to_parquet(f"{self.configs.get_dict()['RAW_DATA_DIR']}/raw_data_{month}.parquet")
        return taxi

    def preprocess(self, df):

        # Basic cleaning
        df = df[df['fare_amount'] > 0].reset_index(drop=True)  # avoid divide-by-zero
        # add target
        df['tip_fraction'] = df['tip_amount'] / df['fare_amount']
        df[self.configs.get_dict()['TARGET_COL']] = df['tip_fraction'] > 0.2

        # add features
        df = synthetic_features(df)

        # drop unused columns
        total_cols = self.configs.get_dict()['FEATURES'] + [self.configs.get_dict()['TARGET_COL']]
        df = df[['tpep_dropoff_datetime'] + self.configs.get_dict()['FEATURES'] +\
                [self.configs.get_dict()['TARGET_COL']]]
        df[total_cols] = df[total_cols].astype("float32").fillna(-1.0)

        # convert target to int32 for efficiency (it's just 0s and 1s)
        df[self.configs.get_dict()['TARGET_COL']] = df[self.configs.get_dict()['TARGET_COL']].astype("int32")

        return df.reset_index(drop=True)

    def get_preprocessed_data(self):
        os.makedirs(self.configs.get_dict()['PROC_DATA_DIR'], exist_ok=True)
        for month in set(self.configs.get_dict()['EVAL_MONTHS'] + [self.configs.get_dict()['TRAINING_MONTH']]):
            raw_data = self.load_raw_data(month)
            pp_data = self.preprocess(raw_data)
            pp_data.to_parquet(f"{self.configs.get_dict()['PROC_DATA_DIR']}/processed_data_{month}.parquet")

