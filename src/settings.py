""" Banking Fraud Detection settings """
import os

PARENT_DIR_PATH = os.path.dirname(os.path.realpath(os.path.join(__file__, '..')))

DATASET_FILENAME = os.path.join(PARENT_DIR_PATH, "data", "banksim_dataset.csv")
MODEL_FILENAME = os.path.join(PARENT_DIR_PATH, "models", "model.pickle")
