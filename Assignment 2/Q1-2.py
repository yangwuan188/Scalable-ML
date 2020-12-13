#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[20]")                            .config("spark.executor.memory",'4G')                            .config("spark.driver.memory",'80G')                                                        .appName("COM6012 assignment 2").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")
data = spark.read.csv('/fastdata/acr19wy/Data/HIGGS.csv')


# In[51]:


num_columns = len(data.columns)
data = data.withColumnRenamed('_c0','label')
column_names = data.schema.names


# In[52]:


from pyspark.sql.types import DoubleType
for i in range(num_columns):
    data = data.withColumn(column_names[i], data[column_names[i]].cast(DoubleType()))


# In[53]:


from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols = column_names[1:num_columns], outputCol = 'features')
raw_plus_vector = assembler.transform(data)
data = raw_plus_vector.select('features','label')


# In[54]:


(trainingData, testData) = data.randomSplit([0.7, 0.3], 20)


# In[55]:


from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
import time
rfc = RandomForestClassifier(labelCol="label", featuresCol="features",numTrees=5,maxDepth=7,maxBins=48)
since = time.time()
rfc_model = rfc.fit(trainingData)
time_elapsed = time.time() - since
print("training time of rfc:",time_elapsed)


# In[56]:


gbt = GBTClassifier(labelCol="label", featuresCol="features",maxDepth=7,maxIter=10,stepSize=0.3)
since = time.time()
gbt_model = gbt.fit(trainingData)
time_elapsed = time.time() - since
print("training time of gbt:",time_elapsed)


# In[57]:


relevant_features_rfc = rfc_model.featureImportances.toArray()
index_sort_rfc = np.argsort(relevant_features_rfc)
index_features_rfc = index_sort_rfc[::-1][:3]
print("most relevant three features of rfc:",index_features_rfc[0]+1,index_features_rfc[1]+1,index_features_rfc[2]+1)


# In[58]:


relevant_features_gbt = gbt_model.featureImportances.toArray()
index_sort_gbt = np.argsort(relevant_features_gbt)
index_features_gbt = index_sort_gbt[::-1][:3]
print("most relevant three features of gbt:",index_features_gbt[0]+1,index_features_gbt[1]+1,index_features_gbt[2]+1)


# In[36]:


prediction_rfc = rfc_model.transform(testData)
prediction_gbt = gbt_model.transform(testData)


# In[37]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
AUC_evaluator = BinaryClassificationEvaluator      (labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
accuracy_evaluator = MulticlassClassificationEvaluator      (labelCol="label", predictionCol="prediction", metricName="accuracy")


# In[38]:


rfc_AUC = AUC_evaluator.evaluate(prediction_rfc)
rfc_accuracy = accuracy_evaluator.evaluate(prediction_rfc)
print("Area Under Curve of rfc:",rfc_AUC)
print("accuracy of rfc:",rfc_accuracy)


# In[39]:


gbt_AUC = AUC_evaluator.evaluate(prediction_gbt)
gbt_accuracy = accuracy_evaluator.evaluate(prediction_gbt)
print("Area Under Curve of gbt:",gbt_AUC)
print("accuracy of gbt:",gbt_accuracy)

