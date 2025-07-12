from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
from src.config import Configurations
import os

class Training():
    def __init__(self):
        self.configs = Configurations()
    
    def change_configs(self, new_configs: dict):
        self.configs.change_configs(new_configs)

    def fit_model(self):
        base_dir = self.configs.get_dict()['PROC_DATA_DIR']
        train_month = self.configs.get_dict()['TRAINING_MONTH']
        taxi_train = pd.read_parquet(f'{base_dir}/processed_data_{train_month}.parquet')
        if self.configs.get_dict()['REDUCE_TRAINING']:
            taxi_train = taxi_train.head(100000)

        model_params = self.configs.get_dict()['MODEL_PARAMETERS']
        rfc = RandomForestClassifier(n_estimators=model_params["n_estimators"], 
                                    max_depth=model_params["max_depth"])
        rfc.fit(taxi_train[self.configs.get_dict()['FEATURES']], 
                taxi_train[self.configs.get_dict()['TARGET_COL']])
        os.makedirs(self.configs.get_dict()['MODEL_DIR'], exist_ok=True)
        joblib.dump(rfc, f"{self.configs.get_dict()['MODEL_DIR']}/random_forest.joblib")
        self.fitted_model = rfc
    
    def get_trained_model(self):
        return self.fitted_model