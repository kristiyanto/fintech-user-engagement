import pandas as pd
from pyspark.sql import functions as f
from .dateHeatmap import date_heatmap


"""
FOR VISUALIZING CALENDAR PLOTS 
"""

def calendarActivity(data, transaction, user_id, title, color="Purples", ax=None):
    all_dates = pd.DataFrame(pd.date_range("2018-01-01", "2019-05-16"), columns=["date"])
    user_activity = (data.where("user_id='%s'" % user_id) #  user_11853, user_15060
                   .select("user_id", "batch_id")
                   .orderBy(f.rand(seed=555)).limit(1)
                   .join(transaction, ["user_id"])
                   .groupBy("user_id", f.col("created_date").cast("date").alias("date"))
                   .agg(f.count("transaction_id").alias("total_transaction"))
                   .toPandas()
                   )
    user_activity["date"] = pd.to_datetime(user_activity.date, format='%Y-%m-%d')

    all_dates = (all_dates.merge(user_activity, on="date", how="left").set_index("date"))

    output = date_heatmap(all_dates["total_transaction"], color=color, ax=ax, edgecolor='grey')
    output.set_title(title)
    return output