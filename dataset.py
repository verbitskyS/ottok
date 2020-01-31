import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import time

warnings.filterwarnings('ignore')


"""
Продготовка данных для леса и регрессии
"""

class Dataset:

    def __init__(self, dir_train='train.csv', one_hot_enc=True):
        self.data_train = pd.read_csv(dir_train)
        self.data_test = None
        self.train_X = None
        self.train_y = None
        self.val_X = None
        self.val_y = None
        self.test_X = None
        self.one_hot_enc = one_hot_enc
        self.dropped_columns = []
        self.dropped_columns_not_important = []
        self.CATEGORICAL_COLUMNS =  None
        self.NUMERIC_COLUMNS = None
        self.dropped_columns_in_train = ['crclscod_ZF', 'crclscod_S', 'crclscod_P1'] #их нет в тесте!!


    def train_prepare(self, cor=0.85, quantile=False, quantile_value=0.999):
        self.corr_porog = cor
        self.data_train = self.data_train.drop(['Customer_ID'], axis=1)

        self.CATEGORICAL_COLUMNS = list(self.data_train.select_dtypes(include='object'))
        self.NUMERIC_COLUMNS = list(self.data_train.select_dtypes(include=['float64', 'int64']))

        if self.one_hot_enc:
            self.data_train = pd.get_dummies(self.data_train, columns=self.CATEGORICAL_COLUMNS)
            self.data_train = self.data_train.drop(self.dropped_columns_in_train, axis=1)
            self.data_train = self.data_train.fillna(self.data_train.median())
        else:
            for column in self.NUMERIC_COLUMNS:
                self.data_train[column] = self.data_train[column].fillna(self.data_train[column].median())
                if quantile:
                    q_high = self.data_train[column].quantile(quantile_value)
                    q_low = self.data_train[column].quantile(1 - quantile_value)
                    self.data_train = self.data_train[
                        (self.data_train[column] <= q_high) & (self.data_train[column] >= q_low)]
            for column in self.CATEGORICAL_COLUMNS:
                self.data_train[column] = self.data_train[column].fillna(self.data_train[column].describe().top)

        """
        Далее убираем сильно зависимые между собой переменные (по одной из пары, где парная между ними корреляция больше self.corr_porog)
        """


        corr_matrix = self.data_train.corr(method='spearman')
        for i in corr_matrix:
            if abs(corr_matrix.drop([i], axis=0)[i].max()) > self.corr_porog:
                self.dropped_columns.append(i)
                corr_matrix = corr_matrix.drop([i], axis=1)
                corr_matrix = corr_matrix.drop([i], axis=0)
        self.data_train = self.data_train.drop(self.dropped_columns, axis=1)
        train, val = train_test_split(self.data_train, test_size=0.2, random_state=42)
        self.train_X = train.drop(['churn'], axis=1)
        self.train_y = train.churn
        self.val_X = val.drop(['churn'], axis=1)
        self.val_y = val.churn
        print(len(train), 'train examples')
        print(len(val), 'validation examples')


    def load_test_data_and_prepare(self, dir_test):
        self.data_test = pd.read_csv(dir_test)
        self.test_X = self.data_test.drop(['Customer_ID'], axis=1)
        self.test_X = self.data_test
        if self.one_hot_enc:
            self.test_X = pd.get_dummies(self.test_X, columns=self.CATEGORICAL_COLUMNS)
            self.test_X = self.data_train.fillna(self.data_train.median())
        else:
            for column in self.NUMERIC_COLUMNS:
                self.test_X[column] = self.test_X[column].fillna(self.test_X[column].median())
            for column in self.CATEGORICAL_COLUMNS:
                self.test_X[column] = self.test_X[column].fillna(self.test_X[column].top())


        # при one-hot enc могут быть разные столбцы в тесте и трейне
        self.test_X = self.test_X.drop(list(set(list(self.train_X)) - set(list(self.test_X))), axis=1)
        self.test_X = self.test_X.drop(list(set(list(self.test_X)) - set(list(self.train_X))),axis =1)


    def save_result(self, probs, name =''):
        pd_result = pd.DataFrame({"Customer_ID": self.data_test.Customer_ID, "churn": probs[:, 1]})
        pd_result.Customer_ID = pd_result.Customer_ID.astype(int)
        pd_result.to_csv('result' + name + str(time.time())+'.csv', index=False)
