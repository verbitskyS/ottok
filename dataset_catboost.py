import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class Dataset:

    def __init__(self, dir_train='train.csv'):
        self.data_train = pd.read_csv(dir_train)
        self.data_test = None
        self.train_X = None
        self.train_y = None
        self.val_X = None
        self.val_y = None
        self.test_X = None
        self.CATEGORICAL_COLUMNS =  None
        self.NUMERIC_COLUMNS = None
        self.cat_feat = None


    def train_prepare(self, quantile=False, quantile_value=0.999):
        self.data_train = self.data_train.drop(['Customer_ID'], axis=1)
        # dataset = dataset.loc[:, most_important_churn] - удаление любых стобцов только все портит
        self.CATEGORICAL_COLUMNS = list(self.data_train.select_dtypes(include='object'))
        self.NUMERIC_COLUMNS = list(self.data_train.select_dtypes(include=['float64', 'int64']))

        for column in self.NUMERIC_COLUMNS:
            self.data_train[column] = self.data_train[column].fillna(self.data_train[column].median())
            # dataset[column]=((dataset[column]-dataset[column].min())/(dataset[column].max()-dataset[column].min()))

        for column in self.CATEGORICAL_COLUMNS:
            self.data_train[column] = self.data_train[column].fillna(self.data_train[column].describe().top)
        dropped_columns = []
        train, val = train_test_split(self.data_train, test_size=0.1, random_state=42)
        for column in self.NUMERIC_COLUMNS:
            if quantile:
                q_high = train[column].quantile(quantile_value)
                q_low = train[column].quantile(1 - quantile_value)
                train = train[(train[column] <= q_high) & (train[column] >= q_low)]
        self.train_X = train.drop(['churn'], axis=1)
        self.train_y = train.churn
        self.val_X = val.drop(['churn'], axis=1)
        self.val_y = val.churn

        #test_X = pd.read_csv('test.csv').drop(['Customer_ID'], axis=1)

        self.cat_feat = list(self.train_X.select_dtypes(include='object'))




    def load_test_data_and_prepare(self, dir_test):
        self.data_test = pd.read_csv(dir_test)
        self.test_X = self.data_test.drop(['Customer_ID'], axis=1)
        self.NUMERIC_COLUMNS = self.NUMERIC_COLUMNS.remove('churn')

        for column in self.NUMERIC_COLUMNS:
            self.test_X[column] = self.test_X[column].fillna(self.test_X[column].median())

        for column in self.CATEGORICAL_COLUMNS:
            self.test_X[column] = self.test_X[column].fillna(self.test_X[column].describe().top)



    def save_result(self, probs, name =''):
        pd_result = pd.DataFrame({"Customer_ID": self.data_test.Customer_ID, "churn": probs[:, 1]})
        pd_result.Customer_ID = pd_result.Customer_ID.astype(int)
        pd_result.to_csv('result' + name + str(time.time())+'.csv', index=False)