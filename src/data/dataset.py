import pandas as pd
from src.config import DATASET_URL, FEATURES, TARGET_COL, RAW_DATA_DIR, PROC_DATA_DIR
from src.features.build_features import synthetic_features
import os 

def load_raw_data():
    taxi = pd.read_parquet(DATASET_URL)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    taxi.to_parquet(f"{RAW_DATA_DIR}/raw_data.parquet")
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
    raw_data = load_raw_data()
    pp_data = preprocess(raw_data)
    os.makedirs(PROC_DATA_DIR, exist_ok=True)
    pp_data.to_parquet(f"{PROC_DATA_DIR}/training_data.parquet")
    return pp_data

if __name__ == "__main__":
    _ = get_preprocessed_data()

