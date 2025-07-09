import pandas as pd
from src.config import DATASET_URL, FEATURES, TARGET_COL, EPS

def load_raw_data():
    taxi = pd.read_parquet(DATASET_URL)
    return taxi

def preprocess(df):

   # Basic cleaning
    df = df[df['fare_amount'] > 0].reset_index(drop=True)  # avoid divide-by-zero
    # add target
    df['tip_fraction'] = df['tip_amount'] / df['fare_amount']
    df[TARGET_COL] = df['tip_fraction'] > 0.2

    # add features
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_minute'] = df['tpep_pickup_datetime'].dt.minute
    df['work_hours'] = (df['pickup_weekday'] >= 0) & (df['pickup_weekday'] <= 4) & (df['pickup_hour'] >= 8) & (df['pickup_hour'] <= 18)
    df['trip_time'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.seconds
    df['trip_speed'] = df['trip_distance'] / (df['trip_time'] + EPS)

    # drop unused columns
    df = df[['tpep_dropoff_datetime'] + FEATURES + [TARGET_COL]]
    df[FEATURES + [TARGET_COL]] = df[FEATURES + [TARGET_COL]].astype("float32").fillna(-1.0)

    # convert target to int32 for efficiency (it's just 0s and 1s)
    df[TARGET_COL] = df[TARGET_COL].astype("int32")

    return df.reset_index(drop=True)

def get_preprocessed_data():
    raw_data = load_raw_data()
    return preprocess(raw_data)

