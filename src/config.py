DATASET_BASE_URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata'

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

MODEL_PARAMETERS = {
    "n_estimators": 100,
    "max_depth": 10,    
}

RAW_DATA_DIR = '../data/raw'
PROC_DATA_DIR = '../data/processed'
MODEL_DIR = '../models'

TRAINING_MONTH = '2020-01'

EVAL_MONTHS = ['2020-01', '2020-02', '2020-03', '2020-05']

REDUCE_TRAINING = False