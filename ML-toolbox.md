# Study notes on machine learning toolboxes on Amazon SageMaker

#### Common tools used in ML projects on SageMaker

## (Updates - 09/02/2020) This document is still being developed - as I'm still learning this tool now! There may be errors in the document!


## PCA analysis
[PCA documentation](https://sagemaker.readthedocs.io/en/latest/pca.html)

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

### 4. 
