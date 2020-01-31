from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
warnings.filterwarnings('ignore')


class Neural:
    def __init__(self, dataset_nn):
        self.data_train = dataset_nn.train_ds
        self.data_val = dataset_nn.val_ds
        self.model = None
        self.predictions_value = None
        self.probs_value = None
        self.feature_columns = dataset_nn.feature_columns

        self.feature_layer = tf.keras.layers.DenseFeatures(self.feature_columns)

        #dropout и BatchNormalization помогают сильно, l2 регулязация нет

        self.model = tf.keras.Sequential([
            self.feature_layer,
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

    def train(self, epochs, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):

        self.model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

        #cal_back = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        #cal_back почему то не работает

        self.model.fit(self.data_train,
                  validation_data=self.data_val,
                  epochs=100)

    def predictions(self, data_test):
        try:
            preds = self.model.predict(data_test)
        except Exception:
            print("модель не натренирована")


    def eval(self):
        try:
            loss, accuracy = self.model.evaluate(self.data_val)
            print("Accuracy", accuracy)
        except Exception:
            print("модель не натренирована")
