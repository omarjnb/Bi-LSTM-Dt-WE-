# All the constants are saved here

# WORD2VEC
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_WORKERS = 8
W2V_MIN_COUNT = 10

# TRAINING PARAMETERS
SEQUENCE_LENGTH = 300
EPOCHS = 20
BATCH_SIZE = 1024
VALIDATION_SPLIT = 0.1
TEST_SIZE = 0.25
VERBOSITY = 0.1

# LSTM HYPERPARAMETERS ASPECT MODELS
DROPOUT_RATE_ASPECT = 0.5
LSTM_UNITS_ASPECT = 100
DROPOUT_ASPECT = 0.3
RECURRENT_DROPOUT_ASPECT = 0.3
DENSE_UNITS_ASPECT_MODEL = 1

# LSTM HYPERPARAMETERS CATEGORY MODELS
DROPOUT_RATE_CATEG = 0.5
LSTM_UNITS_CATEG = 100
DROPOUT_CATEG = 0.3
RECURRENT_DROPOUT_CATEG = 0.3
DENSE_UNITS_CATEGORY_MODEL = 5

# CNN PARAMETERS
FILTERS = 250
KERNEL_SIZE = 3


# COMPILATION PARAMETERS
LOSS_ASPECT = "binary_crossentropy"
LOSS_CATEGORY = "categorical_crossentropy"
OPTIMIZER = "nadam"
#METRICS_LIST = ['accuracy']
METRICS_LIST = ['AUC', 'accuracy', 'Precision', 'Recall']
# CALLBACK PARAMETERS
# ReduceLROnPlateau parameters
REDUCELR_MONITOR = 'val_loss'
REDUCELR_PATIENCE = 5
REDUCELR_COOLDOWN = 0

# EarlyStopping parameters
EARLYSTOP_MONITOR = 'val_acc'
EARLYSTOP_MINDELTA = 1e-4
EARLYSTOP_PATIENCE = 5

import pathlib
# Directory paths to be declared her
HOME_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = HOME_DIR / "data"
TRAIN_DATA_FILE_PATH = DATA_DIR / "Restaurants_Train.xml"
SAVED_MODELS_DIR = HOME_DIR / "models"
SAVE_PIPELINE_PATH = SAVED_MODELS_DIR / "pipeline.pkl"
MODEL_ASPECT_TYPE = SAVED_MODELS_DIR / "model_aspect.h5"
MODEL_CATEGORY = SAVED_MODELS_DIR / "model_category.h5"
# this is the path for encoded aspect classes
ENCODED_ASPECT_PATH = HOME_DIR / "encoded_aspect.json"
# this is the path for encoded category classes
ENCODED_CATEGORY_PATH = HOME_DIR / "encoded_category.json"
