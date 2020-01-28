import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class Dataset:

    def __init__(self, dir_train='train.csv'):
        self.data_train = pd.read_csv(dir_train)
        self.data_test = None
        self.train_X = None
        self.train_y = None
        self.val_X = None
        self.val_y = None
        self.test_X = None
        self.dropped_columns = []

    def prepare(self, cor=0.85, type='train', quantile=False, quantile_value=0.999):
        self.corr_porog = cor
        if type == 'train':
            dataset = self.data_train

        elif type == 'test':
            try:
                dataset = self.data_test
            except Exception:
                print("Data test not found!!!")
        else:
            print("Type Error")


        dataset = dataset.drop(['Customer_ID'], axis=1)

        CATEGORICAL_COLUMNS = list(dataset.select_dtypes(include='object'))
        NUMERIC_COLUMNS = list(dataset.select_dtypes(include=['float64', 'int64']))

        for column in NUMERIC_COLUMNS:
            dataset[column] = dataset[column].fillna(dataset[column].median())
            if quantile and type == 'train':
                q_high = dataset[column].quantile(quantile_value)
                q_low = dataset[column].quantile(1 - quantile_value)
                dataset = dataset[(dataset[column] <= q_high) & (dataset[column] >= q_low)]


        for column in CATEGORICAL_COLUMNS:
            dataset[column] = dataset[column].fillna(dataset[column].describe().top)
            dataset[column] = dataset[column].astype("category").cat.codes

        """
        Далее убираем сильно зависимые между собой переменные (по одной из пары, где парная между ними корреляция больше self.corr_porog)
        """

        if type == 'train':
            corr_matrix = dataset.corr()

            for i in corr_matrix:
                if abs(corr_matrix.drop([i], axis=0)[i].max()) > self.corr_porog:
                    self.dropped_columns.append(i)
                    corr_matrix = corr_matrix.drop([i], axis=1)
                    corr_matrix = corr_matrix.drop([i], axis=0)
            dataset = dataset.drop(self.dropped_columns, axis=1)
            train, val = train_test_split(dataset, test_size=0.1, random_state=42)
            self.train_X = train.drop(['churn'], axis=1)
            self.train_y = train.churn
            self.val_X = val.drop(['churn'], axis=1)
            self.val_y = val.churn
            print(len(train), 'train examples')
            print(len(val), 'validation examples')

        elif 'test':
            dataset = dataset.drop(self.dropped_columns, axis=1)
            self.test_X = dataset
            print(len(self.test_X), 'test examples')
        else:
            print("Type Error!!!")

    def load_test_data_and_prepare(self, dir_test):
        self.data_test = pd.read_csv(dir_test)
        self.prepare(type='test')

    def save_result(self, probs):
        pd_result = pd.DataFrame({"Customer_ID": self.data_test.Customer_ID, "churn": probs[:, 1]})
        pd_result.Customer_ID = pd_result.Customer_ID.astype(int)
        pd_result.to_csv('result.csv', index=False)
