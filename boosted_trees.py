from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
from IPython.display import clear_output
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
# Load dataset.
dataset = pd.read_csv('train.csv').head(100)
dftrain, dfeval = train_test_split(dataset, test_size=0.1, random_state=42)
dftrain = pd.read_csv('train.csv')

y_train = dftrain.pop('churn')
y_eval = dfeval.pop('churn')



import tensorflow as tf
tf.random.set_seed(123)

fc = tf.feature_column


CATEGORICAL_COLUMNS = list(dftrain.select_dtypes(include='object'))
NUMERIC_COLUMNS = list(dftrain.select_dtypes(include=['float64','int64']))


dftrain[NUMERIC_COLUMNS] = dftrain[NUMERIC_COLUMNS].fillna(dftrain[NUMERIC_COLUMNS].mean())
dftrain[CATEGORICAL_COLUMNS] = dftrain[CATEGORICAL_COLUMNS].fillna("Y")
dfeval[NUMERIC_COLUMNS] = dfeval[NUMERIC_COLUMNS].fillna(dfeval[NUMERIC_COLUMNS].mean())
dfeval[CATEGORICAL_COLUMNS] = dfeval[CATEGORICAL_COLUMNS].fillna("Y")



def one_hot_cat_column(feature_name, vocab):
  return tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(feature_name,
                                                 vocab))
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  # Need to one-hot encode categorical features.
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name,
                                           dtype=tf.float32))



#example = dict(dftrain.head(1))
#class_fc = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('class', ('First', 'Second', 'Third')))
#print('Feature value: "{}"'.format(example['mou_Mean'].iloc[0]))
#print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_fc])(example).numpy())

#tf.keras.layers.DenseFeatures(feature_columns)(example).numpy()


# Use entire batch since this is such a small dataset.
NUM_EXAMPLES = len(y_train)

def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
      dataset = dataset.shuffle(NUM_EXAMPLES)
    # For training, cycle thru dataset as many times as need (n_epochs=None).
    dataset = dataset.repeat(n_epochs)
    # In memory training doesn't use batching.
    dataset = dataset.batch(NUM_EXAMPLES)
    return dataset
  return input_fn


print("fssfsdf")
# Training and evaluation input functions.
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)

linear_est = tf.estimator.LinearClassifier(feature_columns)
clear_output()
print("fssfsdf")
# Train model.
linear_est.train(train_input_fn, max_steps=10)
print("fssfsdf")

# Evaluation.
result = linear_est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))

# Since data fits into memory, use entire dataset per layer. It will be faster.
# Above one batch is defined as the entire dataset.
n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns,
                                          n_batches_per_layer=n_batches)

# The model will stop training once the specified number of trees is built, not
# based on the number of steps.
est.train(train_input_fn, max_steps=100)

# Eval.
result = est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))

pred_dicts = list(est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')
plt.show()

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)
plt.show()