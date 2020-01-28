from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression


class Lasso:
    def __init__(self, dataset):
        self.data_train_X = dataset.train_X
        self.data_test_X = dataset.val_X
        self.data_train_y = dataset.train_y
        self.data_test_y = dataset.val_y
        self.model = None
        self.predictions_value = None
        self.probs_value = None

    def train(self):
        self.model = LassoCV(cv=5, random_state=56).fit(self.data_train_X, self.data_train_y)

    def predictions(self, X_test):
        try:
            self.predictions_value = self.model.predict(X_test)
            return self.predictions_value
        except Exception:
            print("Error!")

    def probs(self, X_test):
        try:
            self.probs_value = self.model.predict_proba(X_test)
            return self.probs_value
        except Exception:
            print("Error!")

    def scores_roc(self):
        try:
            pred_val = self.model.predict(self.data_test_X)
            print("Roc val Lasso: " + str(roc_auc_score(self.data_test_y, pred_val)))
        except Exception:
            print("Error!")


