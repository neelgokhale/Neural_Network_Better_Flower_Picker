# Neural_Network_Better_Flower_Picker
An improved version of traditional 2 type, 2 variable linear-regression classifier. This program decides the species of a flower (from 3 different types) based on petal and sepal dimensions.

## Inputs & Setup


```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
from IPython.display import clear_output
import matplotlib.pyplot as plt
```

## Dataset

Iris flowers have 3 species:

* Setosa
* Versicolor
* Virginica

We have the following information about each flower:

* Sepal length
* Sepal width
* Petal length
* Petal width

Example (source: wikipedia)

![Flower](https://upload.wikimedia.org/wikipedia/commons/7/78/Petal-sepal.jpg)


```python
CSV_COLUMN_NAMES = ['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
```


```python
# keras datasets into panda dataframes
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
train.head()

Setosa = {'sepal':[[], []], 'petal':[[], []]}
Versicolor = {'sepal':[[], []], 'petal':[[], []]}
Virginica = {'sepal':[[], []], 'petal':[[], []]}

for i in range(len(train)):
    
    if train['Species'][i] == 0:
        Setosa['sepal'][0].append(train['Sepal_length'][i])
        Setosa['sepal'][1].append(train['Sepal_width'][i])
        Setosa['petal'][0].append(train['Petal_length'][i])
        Setosa['petal'][1].append(train['Petal_width'][i])
    if train['Species'][i] == 1:
        Versicolor['sepal'][0].append(train['Sepal_length'][i])
        Versicolor['sepal'][1].append(train['Sepal_width'][i])
        Versicolor['petal'][0].append(train['Petal_length'][i])
        Versicolor['petal'][1].append(train['Petal_width'][i])
    if train['Species'][i] == 2:
        Virginica['sepal'][0].append(train['Sepal_length'][i])
        Virginica['sepal'][1].append(train['Sepal_width'][i])
        Virginica['petal'][0].append(train['Petal_length'][i])
        Virginica['petal'][1].append(train['Petal_width'][i])

```

## Visualizing Sepal Dimensions


```python
plt.scatter(Setosa['sepal'][0], Setosa['sepal'][1], c='r')
plt.scatter(Versicolor['sepal'][0], Versicolor['sepal'][1], c='b')
plt.scatter(Virginica['sepal'][0], Virginica['sepal'][1], c='g')

plt.legend([SPECIES[0], SPECIES[1], SPECIES[2]], loc='lower right')
plt.xlabel('Sepal_length')
plt.ylabel('Sepal_width')
```




    Text(0, 0.5, 'Sepal_width')




![png](/img/output_8_1.png)


## Visualizing Petal Dimensions


```python
plt.scatter(Setosa['petal'][0], Setosa['petal'][1], c='r')
plt.scatter(Versicolor['petal'][0], Versicolor['petal'][1], c='b')
plt.scatter(Virginica['petal'][0], Virginica['petal'][1], c='g')

plt.legend([SPECIES[0], SPECIES[1], SPECIES[2]], loc='lower right')
plt.xlabel('Petal_length')
plt.ylabel('Petal_width')
```




    Text(0, 0.5, 'Petal_width')




![png](/img/output_10_1.png)



```python
train_y = train.pop('Species')
test_y = test.pop('Species')
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sepal_length</th>
      <th>Sepal_width</th>
      <th>Petal_length</th>
      <th>Petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.4</td>
      <td>2.8</td>
      <td>5.6</td>
      <td>2.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>2.3</td>
      <td>3.3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.9</td>
      <td>2.5</td>
      <td>4.5</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.9</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.7</td>
      <td>3.8</td>
      <td>1.7</td>
      <td>0.3</td>
    </tr>
  </tbody>
</table>
</div>



## Input Function


```python
def input_fn(features, labels, training=True, batch_size=256):
    # convert the inputs into a dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    
    # shuffle and repeat if training is true
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)
```

## Feature Columns


```python
# feature columns describe how to use the input

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

my_feature_columns
```




    [NumericColumn(key='Sepal_length', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
     NumericColumn(key='Sepal_width', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
     NumericColumn(key='Petal_length', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
     NumericColumn(key='Petal_width', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)]



## Building Classification Model

Two choices:

1. `DNNClassifier` Deep Neural Network
2. `LinearClassifier` Linear Regression with Classification


```python
# Build DNN with 2 hidden layers with 30 and 10 hidden nodes each
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # 2 hidder layers of 30, 10 nodes respectivels
    hidden_units=[30, 10],
    # define number of options to choose from
    n_classes=3)
```

    INFO:tensorflow:Using default config.
    WARNING:tensorflow:Using temporary folder as model directory: C:\Users\Owner\AppData\Local\Temp\tmpwfhkcazv
    INFO:tensorflow:Using config: {'_model_dir': 'C:\\Users\\Owner\\AppData\\Local\\Temp\\tmpwfhkcazv', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': ClusterSpec({}), '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
    

## Training


```python
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=10000)
clear_output()
```


```python
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
```

    INFO:tensorflow:Calling model_fn.
    WARNING:tensorflow:Layer dnn is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
    If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
    To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Starting evaluation at 2020-05-19T08:50:57Z
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from C:\Users\Owner\AppData\Local\Temp\tmpwfhkcazv\model.ckpt-10000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Inference Time : 0.43782s
    INFO:tensorflow:Finished evaluation at 2020-05-19-08:50:57
    INFO:tensorflow:Saving dict for global step 10000: accuracy = 0.96666664, average_loss = 0.22104546, global_step = 10000, loss = 0.22104546
    INFO:tensorflow:Saving 'checkpoint_path' summary for global step 10000: C:\Users\Owner\AppData\Local\Temp\tmpwfhkcazv\model.ckpt-10000
    
    Test set accuracy: 0.967
    
    

## Predictions


```python
# Reference data
test_names = []
SPECIES_NAME = {
    0:SPECIES[0],
    1:SPECIES[1],
    2:SPECIES[2]}

for i in test_y:
    test_names.append(SPECIES_NAME[i])
    
test['Species'] = test_names

test.head(test.shape[0])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sepal_length</th>
      <th>Sepal_width</th>
      <th>Petal_length</th>
      <th>Petal_width</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>4.2</td>
      <td>1.5</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.9</td>
      <td>3.1</td>
      <td>5.4</td>
      <td>2.1</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.1</td>
      <td>3.3</td>
      <td>1.7</td>
      <td>0.5</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.0</td>
      <td>3.4</td>
      <td>4.5</td>
      <td>1.6</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.5</td>
      <td>2.5</td>
      <td>4.0</td>
      <td>1.3</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.2</td>
      <td>2.9</td>
      <td>4.3</td>
      <td>1.3</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5.5</td>
      <td>4.2</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6.3</td>
      <td>2.8</td>
      <td>5.1</td>
      <td>1.5</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5.6</td>
      <td>3.0</td>
      <td>4.1</td>
      <td>1.3</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6.7</td>
      <td>2.5</td>
      <td>5.8</td>
      <td>1.8</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7.1</td>
      <td>3.0</td>
      <td>5.9</td>
      <td>2.1</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4.3</td>
      <td>3.0</td>
      <td>1.1</td>
      <td>0.1</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5.6</td>
      <td>2.8</td>
      <td>4.9</td>
      <td>2.0</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5.5</td>
      <td>2.3</td>
      <td>4.0</td>
      <td>1.3</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>14</th>
      <td>6.0</td>
      <td>2.2</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5.7</td>
      <td>2.6</td>
      <td>3.5</td>
      <td>1.0</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>17</th>
      <td>4.8</td>
      <td>3.4</td>
      <td>1.9</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5.1</td>
      <td>3.4</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>19</th>
      <td>5.7</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>20</th>
      <td>5.4</td>
      <td>3.4</td>
      <td>1.7</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>21</th>
      <td>5.6</td>
      <td>3.0</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>22</th>
      <td>6.3</td>
      <td>2.9</td>
      <td>5.6</td>
      <td>1.8</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>23</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>4.9</td>
      <td>1.5</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>24</th>
      <td>5.8</td>
      <td>2.7</td>
      <td>3.9</td>
      <td>1.2</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>25</th>
      <td>6.1</td>
      <td>3.0</td>
      <td>4.6</td>
      <td>1.4</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>26</th>
      <td>5.2</td>
      <td>4.1</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>27</th>
      <td>6.7</td>
      <td>3.1</td>
      <td>4.7</td>
      <td>1.5</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>28</th>
      <td>6.7</td>
      <td>3.3</td>
      <td>5.7</td>
      <td>2.5</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>29</th>
      <td>6.4</td>
      <td>2.9</td>
      <td>4.3</td>
      <td>1.3</td>
      <td>Versicolor</td>
    </tr>
  </tbody>
</table>
</div>




```python
def input_fn(features, batch_size=256):
    # Convert inputs to a dataset without labels
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width']
predict = {}

print("Please type numeric values")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit(): valid = False
    
    predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))

for pred_dict in predictions:
    print(pred_dict)
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    
    print('Prediction is "{}" with {:.1f}% confidence'.format(SPECIES[class_id], 100 * probability))
    
```

    Please type numeric values
    

    Sepal_length:  5.5
    Sepal_width:  6.2
    Petal_length:  1.2
    Petal_width:  1.3
    

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from C:\Users\Owner\AppData\Local\Temp\tmpwfhkcazv\model.ckpt-10000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    {'logits': array([ 6.0373487,  0.9260005, -3.1705947], dtype=float32), 'probabilities': array([9.9390912e-01, 5.9912354e-03, 9.9629484e-05], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1, 2]), 'all_classes': array([b'0', b'1', b'2'], dtype=object)}
    Prediction is "Setosa" with 99.4% confidence
    


```python
plt.scatter(Setosa['sepal'][0], Setosa['sepal'][1], c='r')
plt.scatter(Versicolor['sepal'][0], Versicolor['sepal'][1], c='b')
plt.scatter(Virginica['sepal'][0], Virginica['sepal'][1], c='g')
plt.xlabel('Sepal_length')
plt.ylabel('Sepal_width')

plt.scatter(predict['Sepal_length'], predict['Sepal_width'], c='black')

plt.legend([SPECIES[0], SPECIES[1], SPECIES[2], 'Selection'], loc='best')
```




    <matplotlib.legend.Legend at 0x2294e127d48>




![png](/img/output_24_1.png)



```python
plt.scatter(Setosa['petal'][0], Setosa['petal'][1], c='r')
plt.scatter(Versicolor['petal'][0], Versicolor['petal'][1], c='b')
plt.scatter(Virginica['petal'][0], Virginica['petal'][1], c='g')
plt.xlabel('Petal_length')
plt.ylabel('Petal_width')

plt.scatter(predict['Petal_length'], predict['Petal_width'], c='black')

plt.legend([SPECIES[0], SPECIES[1], SPECIES[2], 'Selection'], loc='best')
```




    <matplotlib.legend.Legend at 0x2294f1739c8>




![png](/img/output_25_1.png)

