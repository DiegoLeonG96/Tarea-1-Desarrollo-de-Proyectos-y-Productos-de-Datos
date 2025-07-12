import pandas as pd
import joblib
from src.config import Configurations
from sklearn.metrics import f1_score

class Evaluation():
    def __init__(self):
        self.configs = Configurations()
    
    def change_configs(self, new_configs: dict):
        self.configs.change_configs(new_configs)

    def eval_model(self):
        loaded_rfc = joblib.load(f"{self.configs.get_dict()['MODEL_DIR']}/random_forest.joblib")
        
        f1_scores = []
        eval_months = self.configs.get_dict()['EVAL_MONTHS']
        eval_months += [self.configs.get_dict()['TRAINING_MONTH']]
        eval_months = list(set(eval_months))
        for month in eval_months:
            eval_data = pd.read_parquet(f"{self.configs.get_dict()['PROC_DATA_DIR']}/processed_data_{month}.parquet")
            preds_test = loaded_rfc.predict_proba(eval_data[self.configs.get_dict()['FEATURES']])
            preds_test_labels = [p[1] for p in preds_test.round()]
            f1_temp = f1_score(eval_data[self.configs.get_dict()['TARGET_COL']], preds_test_labels)
            f1_scores.append(f1_temp)
        f1_df = pd.DataFrame({"month": eval_months, "f1_score": f1_scores})
        f1_df.sort_values(by=['month'], inplace=True)
        
        return f1_df