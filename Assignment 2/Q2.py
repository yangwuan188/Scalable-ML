#!/usr/bin/env python
# coding: utf-8

# In[453]:


import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[20]")                            .config("spark.executor.memory",'3G')                            .config("spark.driver.memory",'60G')                                                       .appName("COM6012 assignment 2").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")
data = spark.read.csv('/fastdata/acr19wy/Data/train_set.csv',header=True)


# In[454]:


data.printSchema()


# In[455]:


schemaNames = data.schema.names
ncolumns = len(schemaNames)


# In[456]:


from pyspark.sql.types import DoubleType
#select features that contribute 
new_data = data.select(schemaNames[2:ncolumns])
new_columns = len(new_data.columns)


# In[457]:


# remove rows with null value
new_data = new_data.dropna()
# remove rows with '?' value
ALLColumns = new_data.columns
expr = ' and '.join('(%s != "?")' % col_name for col_name in ALLColumns)
new_data = new_data.filter(expr)


# In[458]:


column_names = new_data.columns
cat_feature = []
for i in range(19):
    cat_feature.append(column_names[i])
cat_feature.append('NVCat')


# In[459]:


from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
stages = []
for column_name in cat_feature:
    stringIndexer = StringIndexer(inputCol = column_name, outputCol = column_name+'ind')
    stages += [stringIndexer]
#Create the pipeline. Assign the satges list to the pipeline key word stages
pipeline = Pipeline(stages = stages)
#fit the pipeline to data
pipelineModel = pipeline.fit(new_data)
df_ind= pipelineModel.transform(new_data)


# In[460]:


from pyspark.sql.types import DoubleType
feature_ind = []
for name in cat_feature:
    feature_ind.append(name+'ind')
con_feature = ['Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8','NVVar1','NVVar2','NVVar3','NVVar4']
#convert Continuous features to double type
for i in range(len(con_feature)):
    df_ind = df_ind.withColumn(con_feature[i], df_ind[con_feature[i]].cast(DoubleType()))
#convert label to double type
from pyspark.sql.types import DoubleType
df = df_ind.withColumn('Claim_Amount', df_ind['Claim_Amount'].cast(DoubleType()))


# In[461]:


#assign weights to imbalanced class
from pyspark.sql.functions import when
ratio = 0.99
df = df.withColumn('weights', when(df['Claim_Amount'] != 0, ratio).otherwise(1*(1-ratio)))


# In[462]:


cat_con_features = feature_ind + con_feature 
cat_con_features.append('weights')


# In[463]:


from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols = cat_con_features[:], outputCol = 'features') 
df_assembler = assembler.transform(df)
df = df_assembler.select('features','Claim_Amount')


# In[464]:


(trainingData, testData) = df.randomSplit([0.7, 0.3], 40)


# In[465]:


from pyspark.ml.regression import LinearRegression
import time
model = LinearRegression(featuresCol='features', labelCol='Claim_Amount', maxIter=50, elasticNetParam=0.01,regParam=0.5)
since = time.time()
model_linear = model.fit(trainingData)
time_elapsed = time.time() - since
print("training time of lr:",time_elapsed)
predictions = model_linear.transform(testData)


# In[467]:


from pyspark.ml.evaluation import RegressionEvaluator
evaluator_mae = RegressionEvaluator      (labelCol="Claim_Amount", predictionCol="prediction", metricName="mae")
evaluator_mse = RegressionEvaluator      (labelCol="Claim_Amount", predictionCol="prediction", metricName="mse")
mae = evaluator_mae.evaluate(predictions)
mse = evaluator_mse.evaluate(predictions)
print("MAE = %g " % mae)
print("MSE = %g " % mse)


# In[468]:


df_with_label = df.withColumn('label', when(df['Claim_Amount'] != 0, 1).otherwise(0))
df_with_label= df_with_label.withColumn('label', df_with_label['label'].cast(DoubleType()))


# In[470]:


(trainingData_tandem,testData_tandem) = df_with_label.randomSplit([0.7, 0.3], 40)


# In[471]:


from pyspark.ml.classification import LogisticRegression
# train a classifier model
lr =  LogisticRegression(featuresCol='features', labelCol='label',predictionCol='prediction_LR', maxIter=50)
since = time.time()
lr_model = lr.fit(trainingData_tandem)
non_zero_claim = trainingData_tandem.filter(trainingData_tandem['label']==1)
# train gamma regression
from pyspark.ml.regression import GeneralizedLinearRegression
GLM = GeneralizedLinearRegression(featuresCol='features',labelCol='Claim_Amount', predictionCol='prediction_GLM',family='gamma')
gamma_model = GLM.fit(non_zero_claim)
time_elapsed = time.time() - since
print("training time of two model:",time_elapsed)


# In[474]:


prediction_LR = lr_model.transform(testData_tandem)
non_zero_claim_LR = prediction_LR.filter(prediction_LR['prediction_LR']==1)


# In[486]:


predictions_GLM = gamma_model.transform(non_zero_claim_LR)


# In[489]:


evaluator_GLM_mae = RegressionEvaluator      (labelCol="Claim_Amount", predictionCol="prediction_GLM", metricName="mae")
evaluator_GLM_mse = RegressionEvaluator      (labelCol="Claim_Amount", predictionCol="prediction_GLM", metricName="mse")
GLM_mae = evaluator_GLM_mae.evaluate(predictions_GLM)
GLM_mse = evaluator_GLM_mse.evaluate(predictions_GLM)
print("MAE = %g " % GLM_mae)
print("MSE = %g " % GLM_mse)

