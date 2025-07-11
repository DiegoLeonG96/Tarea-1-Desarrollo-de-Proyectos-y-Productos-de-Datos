from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
from src.config import FEATURES, TARGET_COL, MODEL_PARAMETERS, MODEL_DIR, PROC_DATA_DIR
from src.config import TRAINING_MONTH, REDUCE_TRAINING
import os

def fit_model():
    taxi_train = pd.read_parquet(f'{PROC_DATA_DIR}/processed_data_{TRAINING_MONTH}.parquet')
    if REDUCE_TRAINING:
        taxi_train = taxi_train.head(100000)
    rfc = RandomForestClassifier(n_estimators=MODEL_PARAMETERS["n_estimators"], 
                                max_depth=MODEL_PARAMETERS["max_depth"])
    rfc.fit(taxi_train[FEATURES], taxi_train[TARGET_COL])
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rfc, f"{MODEL_DIR}/random_forest.joblib")
    return rfc

if __name__ == '__main__':
    _ = fit_model()