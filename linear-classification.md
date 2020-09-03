# Study notes on linear classification on Amazon SageMaker

#### Common combos used in linear classification ML projects on SageMaker

## (Updates - 09/03/2020) This document is still being developed - as I'm still learning this tool now! There may be errors in the document!


## Create the model

``` python3
from sagemaker import LinearLearner

linear_model = LinearLearner(role = sagemaker.get_execution_role(),
                             train_instance_count = 1,
                             train_instance_type = 'ml.c4.xlarge',
                             predictor_type = 'binary_classifier',
                             output_path = "s3://{}/{}".format(bucket_name, folder_prefix),
                             sagemaker_session = sagemaker.Session(),
                             epochs = 15)
```

``

### Model improvement

Linear model can be improved in 2 ways:

1. **Model tuning** - optimization according to a specific metric.

For example, if we want **recall** (true positives / (true positives + false negatives))
to be at least 90%:

``` python3
linear_model = LinearLearner(...,
                              binary_classifier_model_selection_criteria = 'precision_at_target_recall',
                              target_recall = 0.9)
```

Or, to aim for better **precision** (true positives / (true positives + false positives)) for at least 90%:
``` python3
linear_model = LinearLearner(...,
                              binary_classifier_model_selection_criteria = 'recall_at_target_precision',
                              target_precision  = 0.9)
```

2. **Class imbalance**

To manage class imbalance, we can set the parameter `positive_example_weight_mult` to be `'balanced'`:

``` python3
linear_model = LinearLearner(...,
                              positive_example_weight_mult = 'balanced')
```

## Train the model

**Create the training data**

*Input format*: The input data train_X, train_y should be numpy arrays

Use the `record_set` method to generate the data into RecordSet format. We set the parameters `train` to be train_X, and `labels` to be train_y

``` python3
formatted_train_data = linear_model.record_set(train = train_X, labels = train_y)
```

**training**

Simply use the `fit` method to train the model

``` python3
linear_model.fit(formatted_train_data)  # formatted_train_data is in the RecordSet format
```

## Deploy the model

``` python3


```

## Resources
[LinearLearner documentation](https://sagemaker.readthedocs.io/en/stable/algorithms/linear_learner.html)
[LinearLearner hyperparameters](https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html)
