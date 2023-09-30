import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def train_evaluate_models(x_train, y_train, x_test,y_test, models, hparams):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            hparam = hparams[list(models.keys())[i]]

            # Using Grid Search Cross Validation
            gs = GridSearchCV(model, hparam, cv=3)
            gs.fit(x_train, y_train)

            # model.fit(x_train, y_train) 

            model.set_params(**gs.best_params_) # **means using dictionary values as parameters
            model.fit(x_train, y_train) # Retrain the current model using the best params.

            y_train_pred = model.predict(x_train)

            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
