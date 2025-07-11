import pandas as pd
import joblib
from src.config import MODEL_DIR, EVAL_MONTHS, TARGET_COL, FEATURES, PROC_DATA_DIR
import os
from sklearn.metrics import f1_score

def eval_model():
    loaded_rfc = joblib.load(f"{MODEL_DIR}/random_forest.joblib")
    
    f1_scores = {}
    for month in EVAL_MONTHS:
        eval_data = pd.read_parquet(f"{PROC_DATA_DIR}/processed_data_{month}.parquet")
        preds_test = loaded_rfc.predict_proba(eval_data[FEATURES])
        preds_test_labels = [p[1] for p in preds_test.round()]
        f1_temp = f1_score(eval_data[TARGET_COL], preds_test_labels)
        f1_scores[month] = f1_temp
    
    return f1_scores

if __name__ == '__main__':
    _ = eval_model()