from catboost import CatBoostClassifier
import numpy as np

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import pandas as pd

class cat:

    def __init__(self, dataset, iterations=4500, min_leaf_in_data =300, depth=4, learning_rate=0.03, l2_leaf_reg=10):
        self.data_train_X = dataset.train_X
        self.data_test_X = dataset.val_X
        self.data_train_y = dataset.train_y
        self.data_test_y = dataset.val_y
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.min_leaf_in_data = min_leaf_in_data
        self.model = None
        self.predictions_value = None
        self.probs_value = None

    def train(self, search_params=False):
        if search_params:
            params = {'depth': [4, 7, 10, 15],
                      'learning_rate' : [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12],
                      'l2_leaf_reg': [1,4,9,15], 'iterations': [500]}
            cb = CatBoostClassifier()
            cb_model = GridSearchCV(cb, params, scoring="roc_auc", cv=5)
            cb_model.fit(self.data_train_X, self.data_train_y)
            self.model = cb_model.best_estimator_
            print(cb_model.best_params_)
            print(self.scores_roc())
        else:
            self.model = CatBoostClassifier(task_type="GPU", eval_metric="AUC", depth=self.depth, iterations= 1000,
                                            l2_leaf_reg= self.l2_leaf_reg, learning_rate= self.learning_rate)
            self.model.fit(self.data_train_X, self.data_train_y)



    def predictions(self, X_test):
        self.predictions_value = self.model.predict(X_test)
        return self.predictions_value



    def probs(self, X_test):
        self.probs_value = self.model.predict_proba(X_test)
        return self.probs_value



    def scores_roc(self):
        try:
            print("Roc val Random Forest: " + str(roc_auc_score(self.data_test_y, self.model.predict(self.data_test_X)[:, 1])))
        except Exception:
            print("Error!")