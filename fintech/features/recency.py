from pyspark.sql import functions as f, SparkSession, DataFrame, Window
from pyspark.sql.functions import col, lit, udf
from pyspark import keyword_only
from pyspark.ml.pipeline import Transformer


"""
RECENCY ACTIVITIES
"""


class f_transcByDayOfWeek_ratio(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_transcByDayOfWeek_ratio, self).__init__()
        self.transc_table = transc_table
        self.day_of_week = ["Monday", "Tuesday", "Wednesday",
                            "Thursday", "Friday", "Saturday", "Sunday"]
        self.feature_names = ["user_transactionOn%s_ratio" %
                              i for i in self.day_of_week]

    def _transform(self, X):
        total_transcation = (self.transc_table.groupBy("user_id")
                                 .count()
                                 .withColumnRenamed("count", "total"))

        total_by_day = (self.transc_table
                            .withColumn("user_transactionOn", f.date_format("created_date", "EEEE"))
                            .groupBy("user_id", "user_transactionOn")
                            .count()
                            .groupBy("user_id")
                            .pivot("user_transactionOn")
                            .agg(f.sum("count")))

        ratio = (total_by_day.join(total_transcation, ["user_id"], "left")
                             .select([f.round((col(i) / col("total")), 3)
                                      .alias("user_transactionOn%s_ratio" % i)
                                      for i in total_by_day.drop("user_id").columns]
                                     + ["user_id"])
                             .fillna(0))

        return X.join(ratio, ["user_id"], "left")


class f_transcByTimeOfDay_ratio(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_transcByTimeOfDay_ratio, self).__init__()
        self.transc_table = transc_table
        time_of_day = ["Morning", "Afternoon", "Evening"]
        self.feature_names = ["user_transactionOn%s_ratio"%i for i in time_of_day]

    def _transform(self, X):
        total_transcation = (self.transc_table.groupBy("user_id")
                                 .count()
                                 .withColumnRenamed("count", "total"))

        total_by_day = (self.transc_table
                            .withColumn("transc_hour", f.hour("created_date"))
                            .withColumn("user_transactionOn", f.when(col("transc_hour") < 11, "Morning")
                                                               .otherwise(f.when(col("transc_hour") < 16, "Afternoon")
                                                                           .otherwise("Evening")))
                            .groupBy("user_id", "user_transactionOn")
                            .count()
                            .groupBy("user_id")
                            .pivot("user_transactionOn")
                            .agg(f.sum("count")))

        ratio = (total_by_day.join(total_transcation, ["user_id"], "left")
                             .select([f.round((col(i) / col("total")), 3)
                                      .alias("user_transactionOn%s_ratio" % i)
                                      for i in total_by_day.drop("user_id").columns]
                                     + ["user_id"])
                             .fillna(0))

        return X.join(ratio, ["user_id"], "left")


class f_user_daysSinceLastTransc(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_daysSinceLastTransc, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_daysSinceLastTransc"

    def _transform(self, X):
        latest_date = (self.transc_table
                           .agg(f.max("created_date"))
                           .collect()[0][0])

        last_transction = (self.transc_table.withColumn("rn", f.row_number().over(
            Window.partitionBy("user_id").orderBy(f.desc("created_date"))))
            .where("rn==1")
            .withColumn(self.feature_names,
                        f.datediff(f.lit(latest_date), col("created_date")))
            .select("user_id", self.feature_names))

        return X.join(last_transction, ["user_id"], "left")


class f_user_transcPrevMonth_count(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_transcPrevMonth_count, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_transcPrevMonth_count"

    def _transform(self, X):
        latest_date = (self.transc_table
                           .agg(f.max("created_date"))
                           .collect()[0][0])

        transc_table = (self.transc_table
                            .withColumn("latest_date", f.lit(latest_date).cast("date"))
                            .withColumn("prev_month_end", f.date_add(f.trunc("latest_date", "month"), -1))
                            .withColumn("prev_month_start", f.trunc("prev_month_end", "month"))
                            .where("created_date between prev_month_start and prev_month_end")
                            .groupBy("user_id")
                            .count()
                            .select("user_id", col("count").alias(self.feature_names))
                        )

        return X.join(transc_table, ["user_id"], "left").fillna(0, subset=[self.feature_names])


class f_user_transcCurrMonth_count(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_transcCurrMonth_count, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_transcCurrMonth_count"

    def _transform(self, X):

        end_of_month = (self.transc_table
                            .agg(f.add_months(f.max("created_date"), 1))
                            .collect()[0][0])

        transc_table = (self.transc_table
                            .withColumn("end_of_month", f.date_add(f.trunc(f.lit(end_of_month).cast("date"), "month"), -1))
                            .withColumn("start_of_month", f.trunc("end_of_month", "month"))
                            .where("created_date between start_of_month and end_of_month")
                            .groupBy("user_id")
                            .count()
                            .select("user_id", col("count").alias(self.feature_names)))

        return X.join(transc_table, ["user_id"], "left").fillna(0, subset=[self.feature_names])


class f_user_transcAmountPrevMonth_total(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_transcAmountPrevMonth_total, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_transcAmountPrevMonth_total"

    def _transform(self, X):
        latest_date = (self.transc_table
                           .agg(f.max("created_date"))
                           .collect()[0][0])

        transc_table = (self.transc_table
                            .withColumn("latest_date", f.lit(latest_date).cast("date"))
                            .withColumn("prev_month_end", f.date_add(f.trunc("latest_date", "month"), -1))
                            .withColumn("prev_month_start", f.trunc("prev_month_end", "month"))
                            .where("created_date between prev_month_start and prev_month_end")
                            .groupBy("user_id")
                            .agg(f.round(f.sum("amount_usd"), 3).alias(self.feature_names))
                        )

        return X.join(transc_table, ["user_id"], "left").fillna(0, subset=[self.feature_names])


class f_user_transcAmountCurrMonth_total(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_transcAmountCurrMonth_total, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_transcAmountCurrMonth_total"

    def _transform(self, X):
        end_of_month = (self.transc_table
                            .agg(f.add_months(f.max("created_date"), 1))
                            .collect()[0][0])

        transc_table = (self.transc_table
                            .withColumn("end_of_month", f.date_add(f.trunc(f.lit(end_of_month).cast("date"), "month"), -1))
                            .withColumn("start_of_month", f.trunc("end_of_month", "month"))
                            .where("created_date between start_of_month and end_of_month")
                            .groupBy("user_id")
                            .agg(f.round(f.sum("amount_usd"), 3).alias(self.feature_names))
                        )

        return X.join(transc_table, ["user_id"], "left").fillna(0, subset=[self.feature_names])


class f_user_transcByStatusPrevMonth_count(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_transcByStatusPrevMonth_count, self).__init__()
        self.transc_table = transc_table

        latest_date = (self.transc_table
                           .agg(f.max("created_date"))
                           .collect()[0][0])

        self.status_count = (self.transc_table
                             .withColumn("latest_date", f.lit(latest_date).cast("date"))
                             .withColumn("prev_month_end", f.date_add(f.trunc("latest_date", "month"), -1))
                             .withColumn("prev_month_start", f.trunc("prev_month_end", "month"))
                             .where("created_date between prev_month_start and prev_month_end")
                             .withColumn("state", f.concat_ws("", lit("user_"),
                                                              f.lower(
                                                                  col("transactions_state")),
                                                              lit("PrevMonth_count")))
                             .groupBy("user_id", "state")
                             .agg(f.count("transaction_id").alias("count")))

        self.feature_names = (self.status_count
                                  .select("state")
                                  .distinct()
                                  .orderBy("state")
                                  .toPandas()["state"]
                                  .to_list())

    def _transform(self, X):

        pivot = (self.status_count.groupBy("user_id")
                 .pivot("state")
                 .agg(f.sum("count"))
                 .select(["user_id"] + self.feature_names))

        return X.join(pivot, ["user_id"], "left").fillna(0, subset=self.feature_names)


class f_user_transcByStatusCurrMonth_count(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_transcByStatusCurrMonth_count, self).__init__()
        self.transc_table = transc_table

        end_of_month = (self.transc_table
                            .agg(f.add_months(f.max("created_date"), 1))
                            .collect()[0][0])

        self.status_count = (self.transc_table
                             .withColumn("end_of_month", f.date_add(f.trunc(f.lit(end_of_month).cast("date"), "month"), -1))
                             .withColumn("start_of_month", f.trunc("end_of_month", "month"))
                             .where("created_date between start_of_month and end_of_month")
                             .withColumn("state", f.concat_ws("", lit("user_"),
                                                              f.lower(
                                                                  col("transactions_state")),
                                                              lit("CurrMonth_count")))
                             .groupBy("user_id", "state")
                             .agg(f.count("transaction_id").alias("count")))

        self.feature_names = (self.status_count
                                  .select("state")
                                  .distinct()
                                  .orderBy("state")
                                  .toPandas()["state"]
                                  .to_list())

    def _transform(self, X):

        pivot = (self.status_count.groupBy("user_id")
                 .pivot("state")
                 .agg(f.sum("count"))
                 .select(["user_id"] + self.feature_names))

        return X.join(pivot, ["user_id"], "left").fillna(0, subset=self.feature_names)


class f_user_transcByTypePrevMonth_count(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_transcByTypePrevMonth_count, self).__init__()
        self.transc_table = transc_table

        latest_date = (self.transc_table
                       .agg(f.max("created_date"))
                       .collect()[0][0])

        self.status_count = (self.transc_table
                             .withColumn("latest_date", f.lit(latest_date).cast("date"))
                             .withColumn("prev_month_end", f.date_add(f.trunc("latest_date", "month"), -1))
                             .withColumn("prev_month_start", f.trunc("prev_month_end", "month"))
                             .where("created_date between prev_month_start and prev_month_end")
                             .withColumn("state", f.concat_ws("", lit("user_"), f.lower(col("transactions_type")), lit("PrevMonth_count")))
                             .groupBy("user_id", "state")
                             .agg(f.count("transaction_id").alias("count")))

        self.feature_names = (self.status_count
                                  .select("state")
                                  .distinct()
                                  .orderBy("state")
                                  .toPandas()["state"]
                                  .to_list())

    def _transform(self, X):

        pivot = (self.status_count.groupBy("user_id")
                 .pivot("state")
                 .agg(f.sum("count"))
                 .select(["user_id"] + self.feature_names))

        return X.join(pivot, ["user_id"], "left").fillna(0, subset=self.feature_names)


class f_user_transcByTypeCurrMonth_count(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_transcByTypeCurrMonth_count, self).__init__()
        self.transc_table = transc_table

        end_of_month = (self.transc_table
                            .agg(f.add_months(f.max("created_date"), 1))
                            .collect()[0][0])

        self.status_count = (self.transc_table
                             .withColumn("end_of_month", f.date_add(f.trunc(f.lit(end_of_month).cast("date"), "month"), -1))
                             .withColumn("start_of_month", f.trunc("end_of_month", "month"))
                             .where("created_date between start_of_month and end_of_month")
                             .withColumn("state", f.concat_ws("", lit("user_"), f.lower(col("transactions_type")), lit("CurrMonth_count")))
                             .groupBy("user_id", "state")
                             .agg(f.count("transaction_id").alias("count")))

        self.feature_names = (self.status_count
                                  .select("state")
                                  .distinct()
                                  .orderBy("state")
                                  .toPandas()["state"]
                                  .to_list())

    def _transform(self, X):
        pivot = (self.status_count.groupBy("user_id")
                 .pivot("state")
                 .agg(f.sum("count"))
                 .select(["user_id"] + self.feature_names))

        return X.join(pivot, ["user_id"], "left").fillna(0, subset=self.feature_names)


class f_user_transcByDirectionPrevMonth_count(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_transcByDirectionPrevMonth_count, self).__init__()
        self.transc_table = transc_table

        latest_date = (self.transc_table
                           .agg(f.max("created_date"))
                           .collect()[0][0])

        self.status_count = (self.transc_table
                             .withColumn("latest_date", f.lit(latest_date).cast("date"))
                             .withColumn("prev_month_end", f.date_add(f.trunc("latest_date", "month"), -1))
                             .withColumn("prev_month_start", f.trunc("prev_month_end", "month"))
                             .where("created_date between prev_month_start and prev_month_end")
                             .withColumn("state", f.concat_ws("", lit("user_"), f.lower(col("direction")), lit("PrevMonth_count")))
                             .groupBy("user_id", "state")
                             .agg(f.count("transaction_id").alias("count")))

        self.feature_names = (self.status_count
                                  .select("state")
                                  .distinct()
                                  .orderBy("state")
                                  .toPandas()["state"]
                                  .to_list())

    def _transform(self, X):
        pivot = (self.status_count.groupBy("user_id")
                 .pivot("state")
                 .agg(f.sum("count"))
                 .select(["user_id"] + self.feature_names))

        return X.join(pivot, ["user_id"], "left").fillna(0, subset=self.feature_names)


class f_user_transcByDirectionCurrMonth_count(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_transcByDirectionCurrMonth_count, self).__init__()
        self.transc_table = transc_table

        end_of_month = (self.transc_table
                            .agg(f.add_months(f.max("created_date"), 1))
                            .collect()[0][0])

        self.status_count = (self.transc_table
                             .withColumn("end_of_month", f.date_add(f.trunc(f.lit(end_of_month).cast("date"), "month"), -1))
                             .withColumn("start_of_month", f.trunc("end_of_month", "month"))
                             .where("created_date between start_of_month and end_of_month")
                             .withColumn("state", f.concat_ws("", lit("user_"), f.lower(col("direction")), lit("CurrMonth_count")))
                             .groupBy("user_id", "state")
                             .agg(f.count("transaction_id").alias("count")))

        self.feature_names = (self.status_count
                                  .select("state")
                                  .distinct()
                                  .orderBy("state")
                                  .toPandas()["state"]
                                  .to_list())

    def _transform(self, X):
        pivot = (self.status_count.groupBy("user_id")
                 .pivot("state")
                 .agg(f.sum("count"))
                 .select(["user_id"] + self.feature_names))

        return X.join(pivot, ["user_id"], "left").fillna(0, subset=self.feature_names)


class f_user_merchantTranscLastMonth_count(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_merchantTranscLastMonth_count, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_merchantTranscLastMonth_count"

    def _transform(self, X):
        latest_date = (self.transc_table
                           .agg(f.max("created_date"))
                           .collect()[0][0])

        merchant_count = (self.transc_table
                          .withColumn("latest_date", f.lit(latest_date).cast("date"))
                          .withColumn("prev_month_end", f.date_add(f.trunc("latest_date", "month"), -1))
                          .withColumn("prev_month_start", f.trunc("prev_month_end", "month"))
                          .where("created_date between prev_month_start and prev_month_end and card_presence=False")
                          .groupBy("user_id")
                          .agg(f.count("transaction_id").alias(self.feature_names)))

        return X.join(merchant_count, ["user_id"], "left").fillna(0, subset=[self.feature_names])


class f_user_nonMerchantTranscCurrMonth_count(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_nonMerchantTranscCurrMonth_count, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_nonMerchantTranscCurrMonth_count"

    def _transform(self, X):
        end_of_month = (self.transc_table
                            .agg(f.add_months(f.max("created_date"), 1))
                            .collect()[0][0])

        merchant_count = (self.transc_table
                          .withColumn("end_of_month", f.date_add(f.trunc(f.lit(end_of_month).cast("date"), "month"), -1))
                          .withColumn("start_of_month", f.trunc("end_of_month", "month"))
                          .where("created_date between start_of_month and end_of_month and card_presence=True")
                          .groupBy("user_id")
                          .agg(f.count("transaction_id").alias(self.feature_names)))

        return X.join(merchant_count, ["user_id"], "left").fillna(0, subset=[self.feature_names])


class f_user_merchantTranscLastMonth_total(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_merchantTranscLastMonth_total, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_merchantTranscLastMonth_total"

    def _transform(self, X):
        latest_date = (self.transc_table
                           .agg(f.max("created_date"))
                           .collect()[0][0])

        merchant_count = (self.transc_table
                          .withColumn("latest_date", f.lit(latest_date).cast("date"))
                          .withColumn("prev_month_end", f.date_add(f.trunc("latest_date", "month"), -1))
                          .withColumn("prev_month_start", f.trunc("prev_month_end", "month"))
                          .where("created_date between prev_month_start and prev_month_end and card_presence=False")
                          .groupBy("user_id")
                          .agg(f.sum("transaction_id").alias(self.feature_names)))

        return X.join(merchant_count, ["user_id"], "left").fillna(0, subset=[self.feature_names])


class f_user_nonMerchantTranscCurrMonth_total(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_nonMerchantTranscCurrMonth_total, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_nonMerchantTranscCurrMonth_total"

    def _transform(self, X):
        end_of_month = (self.transc_table
                            .agg(f.add_months(f.max("created_date"), 1))
                            .collect()[0][0])

        merchant_count = (self.transc_table
                          .withColumn("end_of_month", f.date_add(f.trunc(f.lit(end_of_month).cast("date"), "month"), -1))
                          .withColumn("start_of_month", f.trunc("end_of_month", "month"))
                          .where("created_date between start_of_month and end_of_month and card_presence=True")
                          .groupBy("user_id")
                          .agg(f.sum("amount_usd").alias(self.feature_names)))

        return X.join(merchant_count, ["user_id"], "left").fillna(0, subset=[self.feature_names])
