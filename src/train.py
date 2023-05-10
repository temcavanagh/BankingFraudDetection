from __future__ import unicode_literals
import logging
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import recall_score, confusion_matrix, precision_recall_fscore_support

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from settings import DATASET_FILENAME, MODEL_FILENAME

logger = logging.getLogger('training')

def main():
    """ Find best model to fit the dataset and save it into file """
    grid_search = set_grid()
    run_grid_search(grid_search)
    save_search_results(grid_search)


def process_dataset():
    """ Read and split dataset into train and test subsets """
    df = pd.read_csv(DATASET_FILENAME, header=0)
    df = df.drop(['zipcodeOri', 'zipMerchant'], axis=1)
    category_cols = df.select_dtypes(include= ['object']).columns
    for col in category_cols:
        df[col] = df[col].astype('category')
    df[category_cols] = df[category_cols].apply(lambda x: x.cat.codes)
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]
    return train_test_split(X, y, test_size=0.3, random_state=42)

def set_grid():
    """ Create new GridSearch object """
    pipeline = Pipeline([
        (u"clf", RandomForestClassifier(class_weight="balanced", random_state=123)),
    ])
    search_params = {"clf__n_estimators": (10,20,50,100,200,500),
                     "clf__max_depth": (1,2,4,8,12)}
    return GridSearchCV(
        estimator=pipeline,
        param_grid=search_params,
        scoring="recall_macro",
        cv=10,
        n_jobs=-1,
        verbose=3,
    )

def run_grid_search(grid_search, show_evaluation=True):
    """ Run GridSearch and compute evaluation metrics """
    X_train, X_test, y_train, y_test = process_dataset()

    grid_search.fit(X_train, y_train)

    predictions = grid_search.predict(X_test)

    if show_evaluation:
        logger.debug("macro_recall: %s", recall_score(y_test, predictions, average="macro"))
        logger.debug(precision_recall_fscore_support(y_test, predictions))
        logger.debug(confusion_matrix(y_test, predictions))

def save_search_results(grid_search):
    """ Save model """
    joblib.dump(grid_search.best_estimator_, MODEL_FILENAME)

if __name__ == "__main__":
    main()