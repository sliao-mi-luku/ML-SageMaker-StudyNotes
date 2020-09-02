# Study notes on basic commands of Amazon SageMaker

#### Common commands used in SageMaker for the ML project

## (Updates - 09/02/2020) This document is still being developed - as I'm still learning this tool now! There may be errors in the document!


## Get the current SageMaker session
``` python3
import sagemaker
session = sagemaker.Session()
```

## Get the current IAM rule
``` python3
from sagemaker import get_execution_role
role = get_execution_role()
print(role)
```

## Get the default S3 bucket
``` python3
import sagemaker
session = sagemaker.Session()
bucket_name = session.default_bucket()
print(bucket_name)
```

## Convert from ndarray to RecordSet
``` python3
data_recordSet = SageMakerModel.record_set(data_ndarray)
```
