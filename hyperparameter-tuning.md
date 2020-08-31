# Study notes on Hyperparameter Tuning

#### How to tune the hyperparameters of an XGBoost model on Amazon SageMaker?

## (Updates - 08/30/2020) This document is still being developed - as I'm still learning this tool now! There may be errors in the document!


## Prerequisites
Before this step, you should already have built a model on the SageMaker. Please refer to my [study notes of creating a model](https://github.com/sliao-mi-luku/ML-SageMaker-StudyNotes/blob/master/Create-XGBoost-Model.md).

When I was learning this part, I have a XGBoost model with me called `xgb` that can predict the sentiment given a csv text as the input.

## Basics of hyperparameters tuning
To me, hyperparameters tuning is an art of selecting a set of values that can make the prediction performance good magically.

In my practice I'm training a XGBoost model to determine


## Create the model


## Create the hyperparameter tuner

First we need to import some finctions from `sagemaker.tuner`

``` python3
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner
```

We will use the `HyperparameterTuner` function to create a tuner object. And the functions `IntegerParameter` and `ContinuousParameter` will be later used in the method
to specify the whether the hypermeter is discrete or continuous, along with the range of the values we want to tune.

Then we create the tuner object called **xgb_hyperparameter_tuner** by the `HyperparameterTuner` function:

``` python3
xgb_hyperparameter_tuner = HyperparameterTuner(estimator = xgb,
                                                objective_metric_name = 'validation:mlogloss')
```
