#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pyspark

from pyspark.sql import SparkSession

spark = SparkSession.builder     .master("local[2]")     .config("spark.local.dir","/fastdata/acr19wy")     .appName("Assignment2")     .getOrCreate()

sc = spark.sparkContext


###########################  Q2A  ##############################


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row


# In[3]:


lines = spark.read.option("header","true").csv("Data/ratings.csv").rdd


# In[4]:


ratingsRDD = lines.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)


# In[5]:


ratings.show(5)


# In[6]:


(split1,split2,split3) = ratings.randomSplit([0.33,0.33,0.34])


# split traing data and test data


training1 = split2.union(split3)
test1 = split1
training2 = split1.union(split3)
test2 = split2
training3 = split1.union(split2)
test3 = split3

# first ALS


first_als = ALS(maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")

model1_first_als = first_als.fit(training1)
predictions1_first_als = model1_first_als.transform(test1)
rmse1_first_als = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction").evaluate(predictions1_first_als)
mae1_first_als = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction").evaluate(predictions1_first_als)


model2_first_als = first_als.fit(training2)
predictions2_first_als = model2_first_als.transform(test2)
rmse2_first_als = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction").evaluate(predictions2_first_als)
mae2_first_als = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction").evaluate(predictions2_first_als)


model3_first_als = first_als.fit(training3)
predictions3_first_als = model3_first_als.transform(test3)
rmse3_first_als = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction").evaluate(predictions3_first_als)
mae3_first_als = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction").evaluate(predictions3_first_als)

df1 = spark.createDataFrame([(rmse1_first_als,mae1_first_als),(rmse2_first_als,mae2_first_als),(rmse3_first_als,mae3_first_als)], ['RMSE','MAE'])
df1.show()
df1.describe().show()

#second ALS

second_als = ALS(rank=5,maxIter=5, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")

model1_second_als = second_als.fit(training1)
predictions1_second_als = model1_second_als.transform(test1)
rmse1_second_als = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction").evaluate(predictions1_second_als)
mae1_second_als = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction").evaluate(predictions1_second_als)


model2_second_als = second_als.fit(training2)
predictions2_second_als = model2_second_als.transform(test2)
rmse2_second_als = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction").evaluate(predictions2_second_als)
mae2_second_als = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction").evaluate(predictions2_second_als)


model3_second_als = second_als.fit(training3)
predictions3_second_als = model3_second_als.transform(test3)
rmse3_second_als = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction").evaluate(predictions3_second_als)
mae3_second_als = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction").evaluate(predictions3_second_als)

df2 = spark.createDataFrame([(rmse1_second_als,mae1_second_als),(rmse2_second_als,mae2_second_als),(rmse3_second_als,mae3_second_als)], ['RMSE','MAE'])
df2.show()
df2.describe().show()


# third ALS


third_als = ALS(rank=15,maxIter=5, regParam=0.3, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")

model1_third_als = third_als.fit(training1)
predictions1_third_als = model1_third_als.transform(test1)
rmse1_third_als = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction").evaluate(predictions1_third_als)
mae1_third_als = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction").evaluate(predictions1_third_als)


model2_third_als = third_als.fit(training2)
predictions2_third_als = model2_third_als.transform(test2)
rmse2_third_als = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction").evaluate(predictions2_third_als)
mae2_third_als = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction").evaluate(predictions2_third_als)


model3_third_als = third_als.fit(training3)
predictions3_third_als = model3_third_als.transform(test3)
rmse3_third_als = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction").evaluate(predictions3_third_als)
mae3_third_als = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction").evaluate(predictions3_third_als)

df3 = spark.createDataFrame([(rmse1_third_als,mae1_third_als),(rmse2_third_als,mae2_third_als),(rmse3_third_als,mae3_third_als)], ['RMSE','MAE'])
df3.show()
df3.describe().show()

######################      Q2C      #######################

from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
tagscore = spark.read.load("Data/genome-scores.csv",format="csv", inferSchema="true", header="true")
tagname = spark.read.load("Data/genome-tags.csv",format="csv", inferSchema="true", header="true")

#model 1
ItemFactors1=model1_first_als.itemFactors
ItemFactors1_feature=ItemFactors1.rdd.map(lambda r: Row(features=Vectors.dense(r[1]),id=int(r[0]))).toDF(['features','movieId'])
kmeans1 = KMeans(k=25)
model1 = kmeans1.fit(ItemFactors1_feature)
prediction1 = model1.transform(ItemFactors1_feature)
prediction1.groupBy('prediction').count().orderBy('count',ascending=False).show(3)



df_cluster1 = prediction1.groupBy('prediction').count().orderBy('count',ascending=False)
movieid_with_cluster1 = prediction1.join(df_cluster1,'prediction',how='left').orderBy('count',ascending=False).select('prediction','movieID')
No1cluster_movieid1 = movieid_with_cluster1.where(movieid_with_cluster1.prediction==19).select('movieId')
No2cluster_movieid1 = movieid_with_cluster1.where(movieid_with_cluster1.prediction==3).select('movieId')
No3cluster_movieid1 = movieid_with_cluster1.where(movieid_with_cluster1.prediction==18).select('movieId')


No1cluster_movieid_tagid1 = tagscore.join(No1cluster_movieid1,'movieId',how='left_semi').orderBy('movieId','tagId')
top3No1cluster_scores1 = No1cluster_movieid_tagid1.groupBy('tagId').sum('relevance').orderBy('sum(relevance)',ascending=False)
top3No1cluster_scores1.show(3)
tagname.where(tagname.tagId==742).show()
tagname.where(tagname.tagId==646).show()
tagname.where(tagname.tagId==867).show()
print("number of movie when tagId = 742",No1cluster_movieid_tagid1.where(No1cluster_movieid_tagid1.tagId==742).count())
print("number of movie when tagId = 646",No1cluster_movieid_tagid1.where(No1cluster_movieid_tagid1.tagId==646).count())
print("number of movie when tagId = 867",No1cluster_movieid_tagid1.where(No1cluster_movieid_tagid1.tagId==867).count())



No2cluster_movieid_tagid1 = tagscore.join(No2cluster_movieid1,'movieId',how='left_semi').orderBy('movieId','tagId')
top3No2cluster_scores1 = No2cluster_movieid_tagid1.groupBy('tagId').sum('relevance').orderBy('sum(relevance)',ascending=False)
top3No2cluster_scores1.show(3)
tagname.where(tagname.tagId==742).show()
tagname.where(tagname.tagId==646).show()
tagname.where(tagname.tagId==323).show()
print("number of movie when tagId = 742:",No2cluster_movieid_tagid1.where(No2cluster_movieid_tagid1.tagId==742).count())
print("number of movie when tagId = 646:",No2cluster_movieid_tagid1.where(No2cluster_movieid_tagid1.tagId==646).count())
print("number of movie when tagId = 323:",No2cluster_movieid_tagid1.where(No2cluster_movieid_tagid1.tagId==323).count())



No3cluster_movieid_tagid1 = tagscore.join(No3cluster_movieid1,'movieId',how='left_semi').orderBy('movieId','tagId')
top3No3cluster_scores1 = No3cluster_movieid_tagid1.groupBy('tagId').sum('relevance').orderBy('sum(relevance)',ascending=False)
top3No3cluster_scores1.show(3)
tagname.where(tagname.tagId==742).show()
tagname.where(tagname.tagId==807).show()
tagname.where(tagname.tagId==646).show()
print("number of movie when tagId = 742:",No3cluster_movieid_tagid1.where(No3cluster_movieid_tagid1.tagId==742).count())
print("number of movie when tagId = 807:",No3cluster_movieid_tagid1.where(No3cluster_movieid_tagid1.tagId==807).count())
print("number of movie when tagId = 646:",No3cluster_movieid_tagid1.where(No3cluster_movieid_tagid1.tagId==646).count())


#model 2

ItemFactors2=model2_first_als.itemFactors
ItemFactors2_feature=ItemFactors2.rdd.map(lambda r: Row(features=Vectors.dense(r[1]),id=int(r[0]))).toDF(['features','movieId'])
kmeans2 = KMeans(k=25)
model2 = kmeans2.fit(ItemFactors2_feature)
prediction2 = model2.transform(ItemFactors2_feature)
prediction2.groupBy('prediction').count().orderBy('count',ascending=False).show(3)


df_cluster2 = prediction2.groupBy('prediction').count().orderBy('count',ascending=False)
movieid_with_cluster2 = prediction2.join(df_cluster2,'prediction',how='left').orderBy('count',ascending=False).select('prediction','movieID')
No1cluster_movieid2 = movieid_with_cluster2.where(movieid_with_cluster2.prediction==1).select('movieId')
No2cluster_movieid2 = movieid_with_cluster2.where(movieid_with_cluster2.prediction==4).select('movieId')
No3cluster_movieid2 = movieid_with_cluster2.where(movieid_with_cluster2.prediction==5).select('movieId')



No1cluster_movieid_tagid2 = tagscore.join(No1cluster_movieid2,'movieId',how='left_semi').orderBy('movieId','tagId')
top3No1cluster_scores2 = No1cluster_movieid_tagid2.groupBy('tagId').sum('relevance').orderBy('sum(relevance)',ascending=False)
top3No1cluster_scores2.show(3)
tagname.where(tagname.tagId==742).show()
tagname.where(tagname.tagId==646).show()
tagname.where(tagname.tagId==195).show()
print("number of movie when tagId = 742",No1cluster_movieid_tagid2.where(No1cluster_movieid_tagid2.tagId==742).count())
print("number of movie when tagId = 646",No1cluster_movieid_tagid2.where(No1cluster_movieid_tagid2.tagId==646).count())
print("number of movie when tagId = 195",No1cluster_movieid_tagid2.where(No1cluster_movieid_tagid2.tagId==195).count())


No2cluster_movieid_tagid2 = tagscore.join(No2cluster_movieid2,'movieId',how='left_semi').orderBy('movieId','tagId')
top3No2cluster_scores2 = No2cluster_movieid_tagid2.groupBy('tagId').sum('relevance').orderBy('sum(relevance)',ascending=False)
top3No2cluster_scores2.show(3)
tagname.where(tagname.tagId==742).show()
tagname.where(tagname.tagId==646).show()
tagname.where(tagname.tagId==867).show()
print("number of movie when tagId = 742:",No2cluster_movieid_tagid2.where(No2cluster_movieid_tagid2.tagId==742).count())
print("number of movie when tagId = 646:",No2cluster_movieid_tagid2.where(No2cluster_movieid_tagid2.tagId==646).count())
print("number of movie when tagId = 867:",No2cluster_movieid_tagid2.where(No2cluster_movieid_tagid2.tagId==867).count())


No3cluster_movieid_tagid2 = tagscore.join(No3cluster_movieid2,'movieId',how='left_semi').orderBy('movieId','tagId')
top3No3cluster_scores2 = No3cluster_movieid_tagid2.groupBy('tagId').sum('relevance').orderBy('sum(relevance)',ascending=False)
top3No3cluster_scores2.show(3)
tagname.where(tagname.tagId==742).show()
tagname.where(tagname.tagId==270).show()
tagname.where(tagname.tagId==1008).show()
print("number of movie when tagId = 742:",No3cluster_movieid_tagid2.where(No3cluster_movieid_tagid2.tagId==742).count())
print("number of movie when tagId = 270:",No3cluster_movieid_tagid2.where(No3cluster_movieid_tagid2.tagId==270).count())
print("number of movie when tagId = 1008:",No3cluster_movieid_tagid2.where(No3cluster_movieid_tagid2.tagId==1008).count())

# model 3

ItemFactors3=model3_first_als.itemFactors
ItemFactors3_feature=ItemFactors3.rdd.map(lambda r: Row(features=Vectors.dense(r[1]),id=int(r[0]))).toDF(['features','movieId'])
kmeans3 = KMeans(k=25)
model3 = kmeans3.fit(ItemFactors3_feature)
prediction3 = model3.transform(ItemFactors3_feature)
prediction3.groupBy('prediction').count().orderBy('count',ascending=False).show(3)



df_cluster3 = prediction3.groupBy('prediction').count().orderBy('count',ascending=False)
movieid_with_cluster3 = prediction3.join(df_cluster3,'prediction',how='left').orderBy('count',ascending=False).select('prediction','movieID')
No1cluster_movieid3 = movieid_with_cluster3.where(movieid_with_cluster3.prediction==17).select('movieId')
No2cluster_movieid3 = movieid_with_cluster3.where(movieid_with_cluster3.prediction==8).select('movieId')
No3cluster_movieid3 = movieid_with_cluster3.where(movieid_with_cluster3.prediction==5).select('movieId')


No1cluster_movieid_tagid3 = tagscore.join(No1cluster_movieid3,'movieId',how='left_semi').orderBy('movieId','tagId')
top3No1cluster_scores3 = No1cluster_movieid_tagid3.groupBy('tagId').sum('relevance').orderBy('sum(relevance)',ascending=False)
top3No1cluster_scores3.show(3)
tagname.where(tagname.tagId==742).show()
tagname.where(tagname.tagId==646).show()
tagname.where(tagname.tagId==195).show()
print("number of movie when tagId = 742",No1cluster_movieid_tagid3.where(No1cluster_movieid_tagid3.tagId==742).count())
print("number of movie when tagId = 646",No1cluster_movieid_tagid3.where(No1cluster_movieid_tagid3.tagId==646).count())
print("number of movie when tagId = 195",No1cluster_movieid_tagid3.where(No1cluster_movieid_tagid3.tagId==195).count())



No2cluster_movieid_tagid3 = tagscore.join(No2cluster_movieid3,'movieId',how='left_semi').orderBy('movieId','tagId')
top3No2cluster_scores3 = No2cluster_movieid_tagid3.groupBy('tagId').sum('relevance').orderBy('sum(relevance)',ascending=False)
top3No2cluster_scores3.show(3)
tagname.where(tagname.tagId==742).show()
tagname.where(tagname.tagId==972).show()
tagname.where(tagname.tagId==1104).show()
print("number of movie when tagId = 742",No2cluster_movieid_tagid3.where(No2cluster_movieid_tagid3.tagId==742).count())
print("number of movie when tagId = 972",No2cluster_movieid_tagid3.where(No2cluster_movieid_tagid3.tagId==972).count())
print("number of movie when tagId = 1104",No2cluster_movieid_tagid3.where(No2cluster_movieid_tagid3.tagId==1104).count())



No3cluster_movieid_tagid3 = tagscore.join(No3cluster_movieid3,'movieId',how='left_semi').orderBy('movieId','tagId')
top3No3cluster_scores3 = No3cluster_movieid_tagid3.groupBy('tagId').sum('relevance').orderBy('sum(relevance)',ascending=False)
top3No3cluster_scores3.show(3)
tagname.where(tagname.tagId==742).show()
tagname.where(tagname.tagId==646).show()
tagname.where(tagname.tagId==270).show()
print("number of movie when tagId = 742",No3cluster_movieid_tagid3.where(No3cluster_movieid_tagid3.tagId==742).count())
print("number of movie when tagId = 646",No3cluster_movieid_tagid3.where(No3cluster_movieid_tagid3.tagId==646).count())
print("number of movie when tagId = 270",No3cluster_movieid_tagid3.where(No3cluster_movieid_tagid3.tagId==270).count())




