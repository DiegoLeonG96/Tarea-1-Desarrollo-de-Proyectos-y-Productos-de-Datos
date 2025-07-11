import pandas as pd
from src.config import DATASET_BASE_URL, FEATURES, TARGET_COL
from src.config import RAW_DATA_DIR, PROC_DATA_DIR, EVAL_MONTHS, TRAINING_MONTH
from src.features.build_features import synthetic_features
import os 

def load_raw_data(month: str):
    taxi = pd.read_parquet(f"{DATASET_BASE_URL}_{month}.parquet")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    taxi.to_parquet(f"{RAW_DATA_DIR}/raw_data_{month}.parquet")
    return taxi

def preprocess(df):

   # Basic cleaning
    df = df[df['fare_amount'] > 0].reset_index(drop=True)  # avoid divide-by-zero
    # add target
    df['tip_fraction'] = df['tip_amount'] / df['fare_amount']
    df[TARGET_COL] = df['tip_fraction'] > 0.2

    # add features
    df = synthetic_features(df)

    # drop unused columns
    df = df[['tpep_dropoff_datetime'] + FEATURES + [TARGET_COL]]
    df[FEATURES + [TARGET_COL]] = df[FEATURES + [TARGET_COL]].astype("float32").fillna(-1.0)

    # convert target to int32 for efficiency (it's just 0s and 1s)
    df[TARGET_COL] = df[TARGET_COL].astype("int32")

    return df.reset_index(drop=True)

def get_preprocessed_data():
    os.makedirs(PROC_DATA_DIR, exist_ok=True)
    for month in set(EVAL_MONTHS + [TRAINING_MONTH]):
        raw_data = load_raw_data(month)
        pp_data = preprocess(raw_data)
        pp_data.to_parquet(f"{PROC_DATA_DIR}/processed_data_{month}.parquet")
    

if __name__ == "__main__":
    get_preprocessed_data()

