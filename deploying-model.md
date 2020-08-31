# Study notes on Deploying a Model

#### How to deploy an XGBoost model on Amazon SageMaker?

## (Updates - 08/30/2020) This document is still being developed - as I'm still learning this tool now! There may be errors in the document!


## Prerequisites
Before this step, you should already have built a model on the SageMaker. Please refer to my [study notes of creating a model](https://github.com/sliao-mi-luku/ML-SageMaker-StudyNotes/blob/master/Create-XGBoost-Model.md).

When I was learning this part, I have a XGBoost model with me called `xgb` that can predict the sentiment given a csv text as the input.

## Basics of Deployment
Deploying means **creating an endpoint**. An endpoint is a interface that users can send data to for predictions.

## Different methods to deploy a model
There're 2 types endpoints.
The endpoint can be on SageMaker, or it could be a public URL.

**Method 1 - on SageMaker**

The first method is to deploy the model in the SageMaker.

**Method 2 - on a web app URL**

In the second method, we can deploy the model by building up a **web app URL** that can take users' input.
The input will be sent to our model for prediction. Finally we'll send back the prediction to the webpage to present to the users.

The differernce between the 2 methods is that the endpoint of **method 2 can be accessed by anyone** without any permissions.

## Deploy the model on SageMaker

Below is the instruction for deploying the model on SageMaker. There are **High Level** and **Low Level** versions.

Steps | High Level | Low Level |
------------ | ------------- | ------------- |
Set endpoint configurations | X | `session.sagemaker_client.create_endpoint_config(...)` |
Create the endpoint | X | `session.sagemaker_client.create_endpoint(...)` |
Visualize the endpoint creation | X | `session.wait_for_endpoint(...)` |
Serialize input data | X | X |
Send data to endpoint | X | `session.sagemaker_runtime_client.invoke_endpoint(...)` |


### Create an endpoint
It can be done by creating a virtual machine (xgb_predictor) by the `deploy` method.
``` python3
xgb_predictor = xgb.deploy(initial_instance_count = 1, instance_type = 'm1.m4.xlarge')
```
where\
`initial_instance_count` specifies the number of virtual machines to use\
`instance_type` specifies the type of the virtual machines\
When this command is executed, the progress will be displayed by some dashed lines (-----) on the Jupyter Notebook. Once it finishes, a "!" will be displayed.

### Setting parameters
We need to set some parameters of our endpoint.

First, we specify the format of the input that the endpoint is supposed to take. We use the `content_type` method to set the input format, and use the `serializer`
method to set the serializer.
``` python3
xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer
```

``` python3
Y_pred = xgb_predictor.predict(X_test.values).decode('utf-8')
Y_pred = np.fromstring(Y_pred, sep = ',')
```

### (must do!) Delete the endpoint !!!!
Amazon AWS will keep charging you as long as the endpoint is running. So the endpoint must be deleted once you're done with the practice.

```python3
xgb_predictor.delete_endpoint()
```
### (consider to do) Delete the data
You can donsider deleting the data as well to save the space on the disk.


## Deploy the model on a web app

With this method, the model will be deployed on a web app URL that every one can access to and send their own data for predictions.

### Basic steps
step 0 - Fulfill the prerequisites\
step 1 - Create a Lambda function\
step 2 - Create an API\
step 3 - Use the API

**Step 0: fulfill the prerequisites**

**Step 1: create a Lambda function**

1-1 Create a new role for the Lambda function

   <1> In AWS console, select **IAM** under **Security, Identity & Compliance**\
   <2> Select **Roles** on the left, click on **Create role**\
   <3> Select **Lambda**, click on **Next: Permissions**\
   <4> Select **AmazonSageMakerFullAccess**, click on **Next: Tags**\
   <5> Click on **Next: Review**\
   <6> Provide the **Role name**, click on **Create role**\
   <7> Done! The role you created will be list under the **Roles** on the left

1-2 Create the Lambda function

   <1> In AWS console, select **Lambda** under **Compute**\
   <2> Click on **Create a function** on the right\
   <3> Select **Author from scratch**\
   <4> Fill in the **Name** box\
   <5> In the **Runtime** box, choose **Python 3.8**\
   <6> Click to expand **Choose or create an execution role**\
   <7> Select **Use an existing role**\
   <8> In the **Existing role** box, choose the role that was just created in step 1-1\
   <9> Click on **Create function**\
   <10> Done! The lambda function is created. By scrolling down you will see a coding window named `lambda_function`.
   
1-3 Code the Lambda function

   <1> Copy and paste the template below to the `lambda_function`\
   <2> Modify the `EndpointName` in the template, which can be got by the command `xgb_predictor.endpoint`\
   <3> Modify the `vocab` dict in the template, which can be got by the command `print(str(vocabulary))`\
   <4> Click on **Save** on the top right corner
   
1-4 (Optional) Create a test event to test the Lambda function

   <1> Drop down the **Select a test event..** on the top right corner, click on **Configure test events**\
   <2> Select **Create new test event**\
   <3> In the **Event template** box, select **Amazon API Gateway AWS Proxy**\
   <4> Fill in the **Event name** box (ex. testEvent)\
   <5> Look for the **"body"** key on the generated code, replace the default value by your own test input (ex. `"body": "Epic movie!"`)\
   <6> Click on **Create**\
   <7> Click on **Test**, and you can see the result in the **Execution Result** window.
   <8> Done!

Now the Lambda function is created, it's time to create an API (i.e., create an URL for everyone!)
   
**Step 2: create an API**

**Notes** The current verison of Amazon AWS has a completely different interface from the Udacity tutorial. I'm still figuring out the way to create the API interface correctly.

2-1 Create the API interface

   <1> In AWS console, select **API Gateway** under **Networking & Content Delivery**\
   <2> In the **REST API** box, click on **Build**\
   <3> Select **New API** under **Create new API**\
   <4> In the **API name** type in the name you want to have (ex. sentimentAnalysis), click on **Create API**\
   <5> In the **Resources** section, click on **Actions** and click on **Create Method**\
   <6> Select **POST** in the drop-down menu, click on the **V** button\
   <7> On the right hand side, select **Lambda Function** as the **Integration type**\
   <8> Select **Use Lambda Proxy Integration**\
   <9> In the **Lambda Function** box, type in the name of the Lambda function you want to use\
   <10> Click on **Save**, and click on **OK** when being asked about adding the permission\

2-2 Deploy the API

   <1> Click on the **Actions** drop-down menu, click on **Deploy API**\
   <2> In the **Deployment stage** box, select **[New Stage]**\
   <3> In the **Stage name** box, type in **prod** (meaning it's for the production)\
   <4> Click on **Deploy**\
   <5> Done! The URL of the API is shown as **Invoke URL**

**Step 3: use the API**

