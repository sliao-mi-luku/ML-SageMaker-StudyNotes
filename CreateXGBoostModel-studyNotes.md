# XGBoost (Amazon SageMaker) Study Notes

#### How to create an XGBoost model on Amazon SageMaker?

## (Updates - 08/30/2020) This document is still being developed - as I'm still learning this tool now!

## About this file

This file documents how I learned to use XGBoost in my ML projects (the Boston housing dataset and the IMDb datasets),
without any previous experience in Aamazon SageMaker and XGboost! This document is continuously updated and formatted as I was taking the Machine Learning Engineer Nanodegree
in Udacity. Therefore this file can be taken as the study notes from a novice.

## Prerequisites

This document assumes that you already know:
1. How to code some python code
2. How to use the Jupyter Notebook

## Amazon SageMaker


## XGBoost


## The Routines

In a nutshell, the whole procedure consists of:
1. importing necessary packages you'll need
2. download the dataset
3. preprocess the dataset
4. upload the data to S3 service
5. construct a xgboost model
6. set the parameteres
7. training
8. make predictions on the test set

Below is a more detailed blueprint for each steps.

### Step 1 - Set up and import the packages
**1.1** The first step is to import some fundamantal packages into the notebook, which usually includes:
``` python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```
**1.2** Next are the necessary modules for SageMaker:
``` python3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer
```
**1.3** 
``` python3
session = sagemaker.Session()
```
**1.4**
``` python3
role = get_execution_role()
```

### Step 2 - Download the dataset
Next step is to download the dataset. The way to download the dataset depend on the dataset. The best way to do is to google how to download the dataset you're
interested in.

For example, to download the **IMDb dataset**, you can do it by:
``` python3
%mkdir ../data
!wget -O ../data/aclImdb_v1.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -zxf ../data/aclImdb_v1.tar.gz -C ../data
```

There may be different ways to download the same dataset. The method shown above was taught by course that I'm taking.

### Step 3 - Data preprocessing

### Step 4 - Upload the data to S3

### Step 5 - Contruct the XGBoost model
**5.1** Construct the container



``` python3
container = get_image_uri(session.boto_region_name, 'xgboost')
```
**5.2** Construnct the estimator
``` python3
xgb = sagemaker.estimator.Estimator(container,
                                    role,
                                    train_instance_count = 1,
                                    train_instance_type = 'ml.m4.xlarge',
                                    output_path = 's3://{}/{}/output'.format(session.default_bucket(), prefix),
                                    sagemaker_session = session)
```

### Step 6 - Set the hyperparameters
``` python3
xgb.set_hyperparameters(max_depth = 5,
                        eta = 0.2,
                        gamma = 4, 
                        min_child_weight = 6,
                        subsample = 0.8,
                        objective = 'binary:logistic',
                        early_stopping_rounds = 10,
                        num_round = 200)
```

### Step 7 - XGBoost training
**7.1** Define the training and validation data
``` python3
s3_input_train = sagemaker.s3_input(s3_data = train_location,
                                    content_type = 'csv')
s3_input_validation = sagemaker.s3_input(s3_data = val_location,
                                    content_type = 'csv')
```

**7.2** Start training
``` python3
xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})
```
Now just wait for the model to be trained. As the training goes, many information about the training process will be displayed.

### Step 8 - Predict the testing data
**8.1** Create a transformer
``` python3
xgb_transformer = xgb.transformer(instance_count = 1, instance_type = 'ml.m4.xlarge')
```

**8.2** Transform the test data
``` python3
xgb_transformer.transform(test_location, content_type = 'text/csv', split_type = 'Line')

xgb_transformer.wait()
```

**8.3** Move the result from S3 to the current directory
``` linux
!aws s3 cp --recursive $xgb_transformer.output_path $data_dir
```

**8.4** See the performance
Now the predictions are saved in the file named by 'test.csv.out', in the folder `data_dir`.

You can then evaluate the performace by whatever metric you designed for the project.

### Step 9 (Optional) - Clean up the space
You can use the command such as `rm` and `rmdir` to remove anything that are no longer needed.
``` terminal
!rm $data_dir/*
!rmdir $data_dir
!rm $cache_dir/*
!rmdir $cache_dir
```

## Done!

Now you've seen the basic structure of using xgboost on Amazon SageMaker. The code can be easily modified and be used in another project.

You just need to code the following youself:

1. the command to download the dataset
2. the pipeline to process the raw dataset into training, validation and test DataFrames

Hope this document helps.

## Reference of the code
The code in the document was modified from the miniproject of Udacity's Machine Learning Engineer Nanodegree

GitHub: https://github.com/udacity/sagemaker-deployment/blob/master/Mini-Projects/IMDB%20Sentiment%20Analysis%20-%20XGBoost%20(Batch%20Transform).ipynb

