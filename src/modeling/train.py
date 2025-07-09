from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from src.config import FEATURES, TARGET_COL, MODEL_PARAMETERS

def fit_model(taxi_train: pd.DataFrame):
    rfc = RandomForestClassifier(n_estimators=MODEL_PARAMETERS["n_estimators"], 
                                max_depth=MODEL_PARAMETERS["max_depth"])
    rfc.fit(taxi_train[FEATURES], taxi_train[TARGET_COL])
    return rfc