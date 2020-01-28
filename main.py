import dataset
import random_forest
import regression
import lasso

dataset = dataset.Dataset('train.csv')

dataset.prepare(quantile=True)

print(dataset.train_X.head())

random_forest = random_forest.rand_forest(dataset)
random_forest.train()
random_forest.scores_roc()

random_forest.show_feature_important()


#lasso = lasso.Lasso(dataset)
#lasso.train()
#lasso.scores_roc()


#regr = regression.Regression(dataset)
#regr.train()
#regr.scores_roc()


dataset.load_test_data_and_prepare('test.csv')
probs = random_forest.probs(dataset.test_X)
dataset.save_result(probs)
