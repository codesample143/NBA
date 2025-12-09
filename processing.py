import pyspark

from pyspark.sql import SparkSession

"""based on your knowledge of distributed architecture for spark you should understand this"""
spark = SparkSession.builder.appName("ReadCSV").getOrCreate()

df = spark.read.csv("nba_data/csv/game.csv")
df.show(2)