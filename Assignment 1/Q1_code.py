#!/usr/bin/env python
# coding: utf-8

# In[96]:


import pyspark
import matplotlib
import matplotlib.pyplot as plt
import re
from pyspark.sql import SparkSession
from pyspark.sql import functions
from pyspark.sql.functions import regexp_extract
spark = SparkSession.builder     .master("local[2]")     .appName("COM6012 Asignment 1")     .config("spark.local.dir","/fastdata/acr19wy")     .getOrCreate()

sc = spark.sparkContext


# In[2]:


spark


# In[3]:


sc


# In[46]:


logFile=spark.read.text("Data/NASA_access_log_Jul95.gz")
Count = []
first_time_slot = 0
for i in range(0,4):
    t = logFile.filter(logFile.value.contains("Jul/1995:0"+str(i))).count()
    first_time_slot = first_time_slot + t
Count.append(first_time_slot)

second_time_slot = 0
for i in range(4,8):
    t = logFile.filter(logFile.value.contains("Jul/1995:0"+str(i))).count()
    second_time_slot = second_time_slot + t
Count.append(second_time_slot)

third_time_slot1 = 0
third_time_slot2 = 0
for i in range(8,10):
    t = logFile.filter(logFile.value.contains("Jul/1995:0"+str(i))).count()
    third_time_slot1 = third_time_slot1 + t
for i in range(10,12):
    t = logFile.filter(logFile.value.contains("Jul/1995:"+str(i))).count()
    third_time_slot2 = third_time_slot2 + t
Count.append(third_time_slot1+third_time_slot2)

fourth_time_slot = 0
for i in range(12,16):
    t = logFile.filter(logFile.value.contains("Jul/1995:"+str(i))).count()
    fourth_time_slot = fourth_time_slot + t
Count.append(fourth_time_slot)

fifth_time_slot = 0
for i in range(16,20):
    t = logFile.filter(logFile.value.contains("Jul/1995:"+str(i))).count()
    fifth_time_slot = fifth_time_slot + t
Count.append(fifth_time_slot)

sixth_time_slot = 0
for i in range(20,24):
    t = logFile.filter(logFile.value.contains("Jul/1995:"+str(i))).count()
    sixth_time_slot = sixth_time_slot + t
Count.append(sixth_time_slot)


# In[52]:


print("total number of requets in six time-slot:",Count)
total = 0
for i in range(len(Count)):
    total= total+ Count[i]
print("total valid requests in file:",total)


# In[53]:


print("number of missed request log :",logFile.count()-total)


# In[75]:


count = 0
for i in range(1,32):
    day = logFile.filter(logFile.value.contains(str(i)+"/Jul/1995")).count()
    if day != 0:
        count = count + 1 
print("total days in valid requests:",count)


# In[64]:


average = []
for i in range(len(Count)):
    ave = Count[i]/count
    average.append(ave)
print("six average value of time-slot:",average)


# In[70]:


plt.bar(range(len(average)), average)
plt.xticks(range(len(average)), ['0-4', '4-8', '8-12', '12-16', '16-20','20-24'])
plt.show()


# In[143]:


files = logFile.filter(logFile.value.contains(".html"))
file_name = files.select(regexp_extract('value','[^/]*\.html',0).alias('file'))
file_count = file_name.groupBy('file').count()
top_20 = file_count.orderBy('count',ascending=False).show(20)


# In[161]:


name = ['ksc','missions','images','liftoff','mission-sts-71','mission-sts-70','apollo','apollo-13','movies','history','countdown','stsref-toc','winvn','mission-sts-69','apollo-13-info','lc39a','apollo-11','tour','fr','atlantis']
count = [40317,24921,24536,22012,16736,16136,14527,14457,12538,11873,8586,7538,7043,6987,5833,5263,5014,4322,4219,3640]
plt.pie(x=count,labels=name,radius=2,textprops = {'fontsize':10, 'color':'black'},autopct = '%3.2f%%')
plt.show()


# In[1]:





# In[163]:





# In[170]:





# In[ ]:




