DATASET_URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet'

NUMERIC_FEAT = [
    "pickup_weekday",
    "pickup_hour",
    'work_hours',
    "pickup_minute",
    "passenger_count",
    'trip_distance',
    'trip_time',
    'trip_speed'
]

CATEGORICAL_FEAT = [
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
]

FEATURES = NUMERIC_FEAT + CATEGORICAL_FEAT
EPS = 1e-7
TARGET_COL = "high_tip"