from pyspark.sql import functions as f, SparkSession, DataFrame, Window
from pyspark.sql.functions import col, lit, udf
from pyspark import keyword_only
from pyspark.ml.pipeline import Transformer

"""
LOCATION
"""


class f_user_freqLocalTransc_ratio(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_freqLocalTransc_ratio, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_freqLocalTransc_ratio"

    def _transform(self, X):

        transc_table = (self.transc_table
                            .withColumn("merchant_country", f.substring("merchant_country", 0, 2))
                            .select("user_id", "merchant_country"))

        local = (X.select("user_id", "home_country")
                  .join(transc_table.withColumnRenamed("merchant_country", "home_country"), ["user_id", "home_country"])
                  .groupBy("user_id")
                  .agg(f.count("user_id").alias("user_freqLocalTransc")))

        intl = (X.select("user_id", "home_country")
                 .join(transc_table, ["user_id"])
                 .where("merchant_country <> home_country")
                 .groupBy("user_id")
                 .agg(f.count("user_id").alias("user_freqIntlTransc")))

        ratio = (local.join(intl, ["user_id"])
                      .withColumn("user_freqLocalTransc_ratio", f.round(col("user_freqLocalTransc") /
                                  (col("user_freqLocalTransc") + col("user_freqIntlTransc")), 3))
                      .select("user_id", self.feature_names)
                      .fillna(0))

        return X.join(ratio, ["user_id"], "left")


class f_user_freqIntlTransc_ratio(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_freqIntlTransc_ratio, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_freqIntlTransc_ratio"

    def _transform(self, X):
        transc_table = (self.transc_table
                            .withColumn("merchant_country", f.substring("merchant_country", 0, 2))
                            .select("user_id", "merchant_country"))

        local = (X.select("user_id", "home_country")
                  .join(transc_table.withColumnRenamed("merchant_country", "home_country"), ["user_id", "home_country"])
                  .groupBy("user_id")
                  .agg(f.count("user_id").alias("user_freqLocalTransc")))

        intl = (X.select("user_id", "home_country")
                .join(transc_table, ["user_id"])
                .where("merchant_country <> home_country")
                .groupBy("user_id")
                .agg(f.count("user_id").alias("user_freqIntlTransc")))

        ratio = (local.join(intl, ["user_id"])
                      .withColumn("user_freqIntlTransc_ratio", f.round(col("user_freqIntlTransc") /
                                  (col("user_freqLocalTransc") + col("user_freqIntlTransc")), 3))
                      .select("user_id", self.feature_names)
                      .fillna(0))

        return X.join(ratio, ["user_id"], "left")


class user_transcLocation_distCount(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(user_transcLocation_distCount, self).__init__()
        kwargs=self._input_kwargs
        self.transc_table=transc_table
        self.feature_names=user_transcLocation_distCount

    def _transform(self, X):
        transc_table=(self.transc_table
                            .withColumn("merchant_country", f.substring("merchant_country", 0, 2))
                            .select("user_id", "merchant_country"))

        counts=(X.select("user_id", "home_country")
                 .join(transc_table, ["user_id"])
                 .where("merchant_country <> home_country")
                 .groupBy("user_id")
                 .agg(f.count("user_id").alias(self.feature_names))
                 .fillna(0))

        return X.join(counts, ["user_id"], "left")


class f_user_localTranscAmount_avg(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_localTranscAmount_avg, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_localTranscAmount_avg"

    def _transform(self, X):
        transc_table = (self.transc_table
                            .withColumn("merchant_country", f.substring("merchant_country", 0, 2))
                            .select("user_id", "merchant_country", "amount_usd"))

        avg_amount = (X.select("user_id", "home_country")
                       .join(transc_table.withColumnRenamed("merchant_country", "home_country"), ["user_id", "home_country"])
                       .groupBy("user_id")
                       .agg(f.round(f.avg("amount_usd"), 3).alias(self.feature_names))
                       .fillna(0))

        return X.join(avg_amount, ["user_id"], "left")


class f_user_intlTranscAmount_avg(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_intlTranscAmount_avg, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_intlTranscAmount_avg"

    def _transform(self, X):
        transc_table = (self.transc_table
                            .withColumn("merchant_country", f.substring("merchant_country", 0, 2))
                            .select("user_id", "merchant_country", "amount_usd"))

        avg_amount = (X.select("user_id", "home_country")
                       .join(transc_table, ["user_id"])
                       .where("merchant_country <> home_country")
                       .groupBy("user_id")
                       .agg(f.round(f.avg("amount_usd"), 3).alias(self.feature_names))
                       .fillna(0))

        return X.join(avg_amount, ["user_id"], "left")


class f_user_topForeign_country(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_topForeign_country, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_topForeign_country"

    def _transform(self, X):
        transc_table = (self.transc_table
                            .select("user_id", "merchant_country")
                            .groupBy("user_id", "merchant_country")
                            .count()
                            .where("count>0")
                            .withColumn("rn", f.row_number()
                                        .over(Window.partitionBy("user_id").orderBy(f.desc("count"))))
                            .where("rn==1")
                            .select(col("merchant_country").alias(self.feature_names),
                                    "user_id")
                        )

        return X.join(transc_table, ["user_id"], "left")


class f_user_topMerchant_city(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_topMerchant_city, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_topMerchant_city"

    def _transform(self, X):
        transc_table = (self.transc_table
                            .select("user_id", "merchant_city")
                            .groupBy("user_id", "merchant_city")
                            .count()
                            .where("count>0")
                            .withColumn("rn", f.row_number().over(Window.partitionBy("user_id").orderBy(f.desc("count"))))
                            .where("rn==1")
                            .select(col("merchant_city").alias(self.feature_names), "user_id")
                        )

        return X.join(transc_table, ["user_id"], "left")
