import dataset, dataset_nn, dataset_catboost
import random_forest, boosted_trees, regression, lasso
import time
import catboost
import neural_network

"""
Примеры реализации
search_params = False  --- параметры по умолчанию, которые найдены в ноутбуке
"""


dataset = dataset.Dataset('train.csv')
dataset.train_prepare(cor=0.85, quantile=True, quantile_value=0.99)
dataset.load_test_data_and_prepare('test.csv')

regr = regression.Regression(dataset)
regr.train(search_params = False)
regr.scores_roc()
probs = regr.probs(dataset.test_X)
dataset.save_result(probs, 'Regression')


rand_forest = random_forest.Random_forest(dataset)
rand_forest.train()
rand_forest.scores_roc()
probs = rand_forest.probs(dataset.test_X)
dataset.save_result(probs, 'Random_forest')


boost = boosted_trees.Boosted_trees(dataset)
boost.train()
boost.scores_roc()
probs = rand_forest.probs(dataset.test_X)
dataset.save_result(probs, 'Random_forest')

dataset_nn = dataset_nn.Dataset('train.csv')
dataset_nn.train_prepare(quantile=True, quantile_value=0.99,type_cat_columns='emb', batch_size = 100)
dataset_nn.load_test_data_and_prepare('test.csv')
neural = neural_network.Neural(dataset_nn)
neural.train(epochs=100, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
probs = neural.predictions(dataset.data_test)
dataset_nn.save_result(probs)


dataset_cb = dataset_catboost.Dataset('train.csv')
dataset_cb.train_prepare(quantile=True, quantile_value=0.99)
dataset_cb.load_test_data_and_prepare('test.csv')
cat = catboost.cat(dataset_cb)
cat.train(search_params=False)
probs = cat.probs(dataset_cb.data_test)
dataset_cb.save_result(probs)