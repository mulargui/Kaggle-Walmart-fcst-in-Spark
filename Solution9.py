
# Spark python api documentation
# http://spark.apache.org/docs/2.0.0/api/python/index.html

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

if __name__ == "__main__":

    spark = SparkSession\
        .builder\
        .appName("Walmart")\
        .getOrCreate()

    #load data
    dftrain = spark.read.csv("train.csv", header=True, mode="DROPMALFORMED", inferSchema=True)
    dftest = spark.read.csv("test.csv", header=True, mode="DROPMALFORMED", inferSchema=True)

    #Add Seasonality Column based on Date
    dftrain = dftrain.withColumn('Seasonality', weekofyear(dftrain.Date))
    dftest = dftest.withColumn('Seasonality', weekofyear(dftest.Date))

    #Add Year column
    dftrain = dftrain.withColumn('Year', year(dftrain.Date))
    dftest = dftest.withColumn('Year', year(dftest.Date))

    #Add previous year column to the test dataset
    dftest = dftest.withColumn('PYear', dftest.Year - 1)
	   
    #adjust sales in weeks 47, 48 and 49 in 2011 to have same number of sale days in week 49 than in 2012
    #rows affected
    dfchristmas = dftrain.filter((dftrain.Year == 2011) & \
        ((dftrain.Seasonality == 47) | (dftrain.Seasonality == 48) | (dftrain.Seasonality == 49)))

    #keep only those wiht 3 weeks of sales
    tmp = dfchristmas.groupBy(dfchristmas.Store, dfchristmas.Dept).count().filter('count = 3')
    #tmp = dfchristmas.groupBy(dfchristmas.Store, dfchristmas.Dept).count().agg(sum('Weekly_Sales')).filter('count = 3')
    
    dfchristmas = dfchristmas.join(tmp, \
        (dfchristmas.Store == tmp.Store) \
        & (dfchristmas.Dept == tmp.Dept), \
        'inner')\
        .select(dfchristmas.Store, dfchristmas.Dept, dfchristmas.Year, dfchristmas.Seasonality, dfchristmas.Weekly_Sales)

    #adjust sales during christmas campaign
    tmp = dfchristmas.groupBy(tmp.Store, tmp.Dept).agg((sum('Weekly_Sales') / 21).alias('sumWS'))

    dfchristmas = dfchristmas.join(tmp, \
        (dfchristmas.Store == tmp.Store) \
        & (dfchristmas.Dept == tmp.Dept), \
        'inner')\
        .select(dfchristmas.Store, dfchristmas.Dept, dfchristmas.Year, dfchristmas.Seasonality, dfchristmas.Weekly_Sales, tmp.sumWS)

    #weeks 47 and 48 shift part to week 49
    dfchristmas = dfchristmas.withColumn('Weekly_Sales', \
        when(((dfchristmas.Seasonality == 47)|(dfchristmas.Seasonality == 48)),\
            dfchristmas.Weekly_Sales - dfchristmas.sumWS)\
                .otherwise(dfchristmas.Weekly_Sales))	
    dfchristmas = dfchristmas.withColumn('Weekly_Sales', \
        when(((dfchristmas.Seasonality == 49)),\
            dfchristmas.Weekly_Sales + (2 * dfchristmas.sumWS))\
                .otherwise(dfchristmas.Weekly_Sales))	

    #update the training set with the new sales figures
    dftrain = dftrain.join(dfchristmas, \
        (dftrain.Store == dfchristmas.Store) \
        & (dftrain.Dept == dfchristmas.Dept) \
        & (dftrain.Seasonality == dfchristmas.Seasonality) \
        & (dftrain.Year == dfchristmas.Year), \
        'left_outer') \
        .select(dftrain.Store, dftrain.Dept, dftrain.Seasonality, dftrain.Year, dfchristmas.Weekly_Sales)

    #Naive prediction: Weekly_Sales are last year sales
    dfresult = dftest.join(dftrain, \
        (dftest.Store == dftrain.Store) \
        & (dftest.Dept == dftrain.Dept) \
        & (dftest.Seasonality == dftrain.Seasonality) \
        & (dftest.PYear == dftrain.Year),\
        'left_outer')\
        .select(dftest.Store, dftest.Dept,dftest.Date, dftrain.Weekly_Sales)

    #for depts with no data use 0 sales
    dfresult = dfresult.withColumn('Weekly_Sales', \
        when(isnull(dfresult.Weekly_Sales),0)\
            .otherwise(dfresult.Weekly_Sales))	

    #preparing the submission file
    #Adding a column with store_department_date
    dfresult = dfresult.withColumn('Id', \
        concat(dfresult.Store, lit('_'), dfresult.Dept, lit('_'), \
            year(dfresult.Date), lit('-'), month(dfresult.Date), lit('-'), dayofmonth(dfresult.Date)))

    #save results in a csv file
    #dfresult = dfresult.coalesce(1)
    dfresult.select(dfresult.Id,dfresult.Weekly_Sales).coalesce(1).write.option("header", "true").mode('overwrite').csv("Solution9.csv")

    spark.stop()
