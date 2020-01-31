from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')


"""
Продготовка данных для леса и регрессии
"""

class Dataset:

    def __init__(self, dir_train='train.csv', one_hot_enc=True):
        self.data_train = pd.read_csv(dir_train)
        self.data_test = None
        self.train_ds = None
        self.val_ds = None
        self.test_X = None
        self.test_ds = None
        self.one_hot_enc = one_hot_enc
        self.CATEGORICAL_COLUMNS =  None
        self.NUMERIC_COLUMNS = None
        self.feature_columns = None

        #найдены с помощью KFold

        self.most_important_churn = ['eqpdays', 'hnd_price', 'hnd_webcap', 'totmrc_Mean', 'asl_flag', 'crclscod', 'mou_Mean',
                                'mou_cvce_Mean',
                                'complete_Mean', 'comp_vce_Mean', 'mou_opkv_Mean', 'avg3mou', 'attempt_Mean',
                                'plcd_vce_Mean', 'peak_vce_Mean', 'opk_vce_Mean',
                                'mou_peav_Mean', 'mou_rvce_Mean', 'models', 'avg3qty', 'owylis_vce_Mean', 'area',
                                'avg6mou', 'recv_vce_Mean', 'ethnic', 'phones', 'uniqsubs',
                                'iwylis_vce_Mean', 'lor', 'avg6qty', 'unan_vce_Mean', 'ccrndmou_Mean',
                                'mouowylisv_Mean', 'custcare_Mean', 'inonemin_Mean',
                                'cc_mou_Mean', 'mouiwylisv_Mean', 'callwait_Mean', 'refurb_new', 'dualband',
                                'change_mou', 'threeway_Mean', 'avgmou', 'churn']

        self.most_important = self.most_important_churn.drop(['churn'], axis =1)

    def df_to_dataset(self, dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        print(dataframe.head())
        labels = dataframe.pop('churn')
        print(dataframe.head())
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds



    def train_prepare(self, quantile=False, quantile_value=0.999, type_cat_columns='emb', batch_size = 100):


        self.data_train = self.data_train.loc[:, self.most_important_churn]


        self.CATEGORICAL_COLUMNS = list(self.data_train.select_dtypes(include='object'))
        self.NUMERIC_COLUMNS = list(self.data_train.select_dtypes(include=['float64', 'int64']))

        for column in self.NUMERIC_COLUMNS:
            self.data_train[column] = self.data_train[column].fillna(self.data_train[column].median())
            if quantile:
                q_high = self.data_train[column].quantile(quantile_value)
                q_low = self.data_train[column].quantile(1 - quantile_value)
                self.data_train = self.data_train[(self.data_train[column] <= q_high) & (self.data_train[column] >= q_low)]

            #стандартизация!!
            self.data_train[column] = ((self.data_train[column] - self.data_train[column].min()) / (
                        self.data_train[column].max() - self.data_train[column].min()))

            # dataframe[column] = (dataframe[column] - dataframe[column].mean())/(dataframe[column].std())

        for column in self.CATEGORICAL_COLUMNS:
            self.data_train[column] = self.data_train[column].fillna(self.data_train[column].describe().top)

        train, test = train_test_split(self.data_train, test_size=0.2)
        train, val = train_test_split(train, test_size=0.2)
        print(len(train), 'train examples')
        print(len(val), 'validation examples')
        print(len(test), 'test examples')

        feature_columns = []

        self.NUMERIC_COLUMNS = self.NUMERIC_COLUMNS[:-1]  #без churn


        for header in self.NUMERIC_COLUMNS:
            feature_columns.append(feature_column.numeric_column(header))

        # cat_col_tf = []

        if type_cat_columns == 'one_hot':
            for col in self.CATEGORICAL_COLUMNS:
                thal = feature_column.categorical_column_with_vocabulary_list(
                    col, self.data_train[col].unique())
                thal_one_hot = feature_column.indicator_column(thal)
                feature_columns.append(thal_one_hot)

        elif 'emb':
            for col in self.CATEGORICAL_COLUMNS:
                vocabulary = self.data_train[col].unique()
                cat_c = tf.feature_column.categorical_column_with_vocabulary_list(col, vocabulary)
                embeding = feature_column.embedding_column(cat_c, dimension=50)
                feature_columns.append(embeding)

        self.train_ds = self.df_to_dataset(train, batch_size=batch_size)
        self.val_ds = self.df_to_dataset(val, shuffle=False, batch_size=batch_size)

    def load_test_data_and_prepare(self, dir_test):
        self.data_test = pd.read_csv(dir_test)
        self.X_test = self.data_test.drop(['Customer_ID'], axis=1)
        self.X_test = self.X_test.loc[:, self.most_important]

        for column in self.NUMERIC_COLUMNS:
            self.X_test[column] = self.X_test[column].fillna(self.X_test[column].median())
            self.X_test[column] = ((self.X_test[column] - self.X_test[column].min()) / (self.X_test[column].max() - self.X_test[column].min()))

        for column in self.CATEGORICAL_COLUMNS:
            self.X_test[column] = self.X_test[column].fillna(self.X_test[column].describe().top)

        self.test_ds = tf.data.Dataset.from_tensor_slices((dict(self.X_test))).batch(batch_size=200)


    def save_result(self, probs):
        pd_result = pd.DataFrame({"Customer_ID": self.data_test.Customer_ID, "churn": probs[:, 1]})
        pd_result.Customer_ID = pd_result.Customer_ID.astype(int)
        pd_result.to_csv('result_NN' +'.csv', index=False)
