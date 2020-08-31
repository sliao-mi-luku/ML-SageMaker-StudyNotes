# Study notes on Hyperparameter Tuning

#### How to tune the hyperparameters of an XGBoost model on Amazon SageMaker?

## (Updates - 08/30/2020) This document is still being developed - as I'm still learning this tool now! There may be errors in the document!


## Prerequisites
Before this step, you should already have built a model on the SageMaker. Please refer to my [study notes of creating a model](https://github.com/sliao-mi-luku/ML-SageMaker-StudyNotes/blob/master/Create-XGBoost-Model.md).

When I was learning this part, I have a XGBoost model with me called `xgb` that can predict the sentiment given a csv text as the input.

## Basics of hyperparameters tuning
