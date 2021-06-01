from pyspark.sql import functions as f, SparkSession, DataFrame, Window
from pyspark.sql.functions import col, lit, udf
from pyspark import keyword_only
from pyspark.ml.pipeline import Transformer


"""
ACTIVITY
"""


class f_user_transcAmount_total(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_transcAmount_total, self)
        self.transc_table = transc_table
        self.feature_names = "user_transcAmount_total"

    def _transform(self, X):
        transc = (self.transc_table
                  .groupBy("user_id")
                  .agg(f.max("amount_usd").alias("total"))
                  .select("user_id", f.round("total", 3).alias(self.feature_names)))

        return X.join(transc, ["user_id"], "left")


class f_user_transcAmount_max(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_transcAmount_max, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_transcAmount_max"

    def _transform(self, X):
        transc = (self.transc_table
                  .groupBy("user_id")
                  .agg(f.max("amount_usd").alias("max"))
                  .select("user_id", f.round("max", 3).alias(self.feature_names)))

        return X.join(transc, ["user_id"], "left")


class f_user_transcAmount_min(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_transcAmount_min, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_transcAmount_min"

    def _transform(self, X):
        transc = (self.transc_table
                  .groupBy("user_id")
                  .agg(f.min("amount_usd").alias("min"))
                  .select("user_id", f.round("min", 3).alias(self.feature_names)))

        return X.join(transc, ["user_id"], "left")


class f_user_transcAmount_avg(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_transcAmount_avg, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_transcAmount_avg"

    def _transform(self, X):
        transc = (self.transc_table
                  .groupBy("user_id")
                  .agg(f.avg("amount_usd").alias("avg"))
                  .select("user_id", f.round("avg", 3).alias(self.feature_names)))

        return X.join(transc, ["user_id"], "left")
