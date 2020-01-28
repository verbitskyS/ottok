from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import pandas as pd

class rand_forest:

    def __init__(self, dataset, n_estimator=450, max_depth=450, min_sample_split=2, min_sample_leaf=1, criterion='entropy'):
        self.data_train_X = dataset.train_X
        self.data_test_X = dataset.val_X
        self.data_train_y = dataset.train_y
        self.data_test_y = dataset.val_y
        self.n_estimator = n_estimator
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.model = None
        self.predictions_value = None
        self.probs_value = None

    def train(self, search_params=False):
        if search_params:
            """
            Находим наилучшие параметры в 3 захода
            criterion='entropy' - лучшее чем gini, проверено в jupyter Notebook, как и дефолтные другие лучшие параметр
            """
            clf_rand = RandomForestClassifier()

            params = {'n_estimators': range(1, 1000, 100), 'max_depth': range(1, 1000, 100)}
            grid_search_random_forest = RandomizedSearchCV(clf_rand, params, cv=5, scoring='accuracy', verbose=10, random_state=42)
            grid_search_random_forest.fit(self.data_train_X, self.data_train_y)
            best_params = grid_search_random_forest.best_params_

            params = {'n_estimators': range(best_params[0]-50, best_params[0]+50, 10), 'max_depth': range(best_params[1]-50, best_params[1]+50, 10)}
            grid_search_random_forest = RandomizedSearchCV(clf_rand, params, cv=5, scoring='accuracy', verbose=10)
            grid_search_random_forest.fit(self.data_train_X, self.data_train_y, random_state=42)
            best_params = grid_search_random_forest.best_params_

            params = {'n_estimators': range(best_params[0] - 10, best_params[0] + 10, 2),
                  'max_depth': range(best_params[1] - 10, best_params[1] + 10, 2)}
            grid_search_random_forest = RandomizedSearchCV(clf_rand, params, cv=5, scoring='accuracy', verbose=10)
            grid_search_random_forest.fit(self.data_train_X, self.data_train_y, random_state=42)
            print('best_params: ' +  str(grid_search_random_forest.best_params_))
            self.n_estimator = grid_search_random_forest.best_params_[0]
            self.max_depth = grid_search_random_forest.best_params_[1]
            self._model = grid_search_random_forest.best_estimator_
        else:
            self.model = RandomForestClassifier(n_estimators=self.n_estimator, criterion=self.criterion,
                                                     max_depth=self.max_depth, min_samples_leaf=self.min_sample_leaf,
                                                     min_samples_split=self.min_sample_split, random_state=42)
            self.model.fit(self.data_train_X, self.data_train_y)


    def show_feature_important(self):
        print(pd.DataFrame(
            {'features': self.data_train_X.columns, 'feature_important': self.model.feature_importances_}).sort_values(
            'feature_important'))


    def predictions(self, X_test):
        self.predictions_value = self.model.predict(X_test)
        return self.predictions_value



    def probs(self, X_test):
        self.probs_value = self.model.predict_proba(X_test)
        return self.probs_value



    def scores_roc(self):
        try:
            pred_val = self.model.predict(self.data_test_X)
            print("Roc val Random Forest: " + str(roc_auc_score(self.data_test_y, pred_val)))
        except Exception:
            print("Error!")



