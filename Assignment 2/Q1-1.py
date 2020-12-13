#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[10]")                            .config("spark.executor.memory",'4G')                            .config("spark.driver.memory",'40G')                                                        .appName("COM6012 assignment 2").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")
data = spark.read.csv('./Data/HIGGS.csv.gz')


# In[12]:


num_columns = len(data.columns)
data = data.withColumnRenamed('_c0','label')
column_names = data.schema.names


# In[13]:


from pyspark.sql.types import DoubleType
for i in range(num_columns):
    data = data.withColumn(column_names[i], data[column_names[i]].cast(DoubleType()))


# In[14]:


data.printSchema()


# In[15]:


sample_data = data.sample(False,0.05,seed=10)


# In[16]:


#split training data and test data 
(trainingData, testData) = sample_data.randomSplit([0.7, 0.3], 20)


# In[17]:


# tuning RandomForest param
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols = column_names[1:num_columns], outputCol = 'features')
rfc = RandomForestClassifier(labelCol="label", featuresCol=assembler.getOutputCol())
rfc_pipeline = Pipeline(stages=[assembler,rfc])


# In[18]:


rfc_paramGrid = ParamGridBuilder()    .addGrid(rfc.maxDepth,[5,6,7])    .addGrid(rfc.numTrees, [3,4,5])    .addGrid(rfc.maxBins, [32,16,48])    .build()


# In[19]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
AUC_evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='label', metricName="areaUnderROC")
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")


# In[20]:


rfc_crossval = CrossValidator(estimator=rfc_pipeline,
                              estimatorParamMaps=rfc_paramGrid,
                              evaluator=AUC_evaluator,
                              numFolds=3)


# In[ ]:


cvModel_rfc = rfc_crossval.fit(trainingData)   


# In[15]:


#get best model
BestPipeline_rfc = cvModel_rfc.bestModel
BestModel_rfc = BestPipeline_rfc.stages[1]


# In[16]:


numTrees = BestModel_rfc.getOrDefault('numTrees')
maxDepth = BestModel_rfc.getOrDefault('maxDepth')
maxBins = BestModel_rfc.getOrDefault('maxBins')
print("param numTrees of rfc:",numTrees)
print("param maxDepth of rfc:",maxDepth)
print("param maxBins of rfc:",maxBins)


# In[30]:


predction_rfc = cvModel_rfc.transform(testData)
AUC_rfc = AUC_evaluator.evaluate(predction_rfc)
print("area under curve of rfc:",AUC_rfc)


# In[ ]:


gbt = GBTClassifier(featuresCol='features', labelCol='label')
gbt_pipeline = Pipeline(stages=[assembler,gbt])


# In[ ]:


gbt_paramGrid = ParamGridBuilder()     .addGrid(gbt.maxIter, [10,5,3])     .addGrid(gbt.stepSize, [0.01,0.1,0.3])     .addGrid(gbt.maxDepth, [5,6,7])     .build()


# In[ ]:


gbt_crossval = CrossValidator(estimator=gbt_pipeline,
                              estimatorParamMaps=gbt_paramGrid,
                              evaluator=AUC_evaluator,
                              numFolds=3)


# In[ ]:


cvModel_gbt = gbt_crossval.fit(trainingData)  


# In[ ]:


BestPipeline_gbt = cvModel_gbt.bestModel
BestModel_gbt = BestPipeline_gbt.stages[1]


# In[ ]:


maxIter = BestModel_gbt.getOrDefault('maxIter')
stepSize = BestModel_gbt.getOrDefault('stepSize')
maxDepth = BestModel_gbt.getOrDefault('maxDepth')
print("param maxIter of gbt:",maxIter)
print("param stepSize of gbt:",stepSize)
print("param maxDepth of gbt:",maxDepth)


# In[ ]:


predction_gbt = cvModel_gbt.transform(testData)
AUC_gbt = AUC_evaluator.evaluate(predction_gbt)
print("area under curve of rfc:",AUC_gbt)

