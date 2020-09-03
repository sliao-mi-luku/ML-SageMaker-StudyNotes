# Study notes on unsupervised learning (PCA + K-Means) on Amazon SageMaker

#### Common combos used in unsupervised learning ML projects on SageMaker

## (Updates - 09/02/2020) This document is still being developed - as I'm still learning this tool now! There may be errors in the document!

---
Many unsupervised learning problems' goals are to cluster the training into groups, and ask what feature(s) determines (or distinguishes) the group that a sample belongs to. Once a model is built, we can label any new data to see which group (cluster) it belongs.

In this kind of problems, a good first try is to use PCA + K-Means.

PCA (principle component analysis) reduces the dimension of the data features, and K-means is used to cluster the data into different groups.

This study notes summarizes how to perform PCA and K-Mean analyses on SageMaker

### Things to know before you start
1. **Additional fees may be incurred** for any service you used on Amazon AWS, such as running a notebook instance or deployment an endpoint.
To minimize the cost, you should **stop the notebook instances** when you're not using it, and **delete the endpoints** whenever you're not using it.
2. This is my study notes when I'm taking *Udacity's Machine Learning Engineer Nanodegree*. The code blocks are extracted from their [Tutorial GitHub](https://github.com/udacity/ML_SageMaker_Studies/blob/master/Population_Segmentation/Pop_Segmentation_Solution.ipynb).

## PCA analysis
SageMaker's [PCA documentation](https://sagemaker.readthedocs.io/en/latest/pca.html)

### 1. Create a PCA model by SageMaker

We create a PCA model and name it `PCA_mode`

``` python3
PCA_model = sagemaker.PCA(role = role,
                          train_instance_count = 1, 
                          train_instance_type = 'ml.c4.xlarge',
                          output_path = output_path,
                          num_components = num_components,
                          sagemaker_session = session)
```

### 2. Convert the data (usually DataFrame) into the RecordSet format

Assume the data (`df`) you have is in pandas DataFrame format. We need to convert it into **RecordSet** format.

We convert `df` into a ndarray `data_ndarray` and then convert it into a RecordSet `data_recordset`

*Why RecordSet?*

RecordSet is required for training algorithms of all SageMaker built-in mcahine learning models

``` python3
data_ndarray = df.values.astype('float32')
data_recordset = PCA_model.record_set(data_ndarray)
```

### 3. Train
``` python3
%%time # magic command to display the timestamps

PCA_model.fit(data_recordset)
```

When the training completes, the result will be listed in the SageMaker subsection **Training jobs** with a job name.

You need to go to the Training subsection and copy the job name that SageMaker created for you.

### 4. Deploy the model to test on a dataset

#### (Caution) Deploying the model will create an endpoint on SageMaker, which will be costing you as long as it's running!
#### Remember to delete the endpoint after practice to stop being charged.

We deploy the model with the name `PCA_predictor` on SageMaker. And we pass the training data again into the predictor to see the outputs.

``` python3
PCA_predictor = PCA_model.deploy(initla_instance_count = 1, instance_type = 'ml.t2.medium')
```
### 5. Delete the endpoint

``` python3
session.delete_endpoint(PCA_predictor.endpoint)
```

