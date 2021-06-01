from pyspark.sql import functions as f, SparkSession, DataFrame, Window
from pyspark.sql.functions import col, lit, udf
from pyspark import keyword_only
from pyspark.ml.pipeline import Transformer

"""
PREFERENCES AND DEMOGRAPHIC
"""


class f_user_device(Transformer):

    @keyword_only
    def __init__(self, **kwargs):
        super(f_user_device, self).__init__()
        self.feature_names = "user_device"

    def _transform(self, X):
        assert 'device' in X.columns
        return X.withColumnRenamed('device', self.feature_names)


class f_user_contact_count(Transformer):

    @keyword_only
    def __init__(self, **kwargs):
        super(f_user_contact_count, self).__init__()
        self.feature_names = "user_contact_count"

    def _transform(self, X):
        return X.withColumn(self.feature_names, col("num_contacts"))


class f_user_allowPush(Transformer):

    @keyword_only
    def __init__(self, **kwargs):
        super(f_user_allowPush, self).__init__()
        self.feature_names = "user_allowPush"

    def _transform(self, X):
        return (X.withColumn(self.feature_names, col("attributes_notifications_marketing_push"))
                 .fillna(0, subset=[self.feature_names]))


class f_user_allowEmail(Transformer):

    @keyword_only
    def __init__(self, **kwargs):
        super(f_user_allowEmail, self).__init__()
        self.feature_names = "user_allowEmail"

    def _transform(self, X):
        return (X.withColumn(self.feature_names, col("attributes_notifications_marketing_email"))
                 .fillna(0, subset=[self.feature_names]))


class f_user_cyptoUnlocked(Transformer):

    @keyword_only
    def __init__(self, **kwargs):
        super(f_user_cyptoUnlocked, self).__init__()
        self.feature_names = "user_cyptoUnlocked"

    def _transform(self, X):
        return X.withColumn(self.feature_names, col("user_settings_crypto_unlocked"))


class f_user_plan(Transformer):

    @keyword_only
    def __init__(self, **kwargs):
        super(f_user_plan, self).__init__()
        self.feature_names = "user_plan"

    def _transform(self, X):
        return X.withColumn(self.feature_names, col("plan"))


class f_user_homeCountry(Transformer):

    @keyword_only
    def __init__(self, **kwargs):
        super(f_user_homeCountry, self).__init__()
        self.feature_names = "user_homeCountry"

    def _transform(self, X):
        return X.withColumn(self.feature_names, col("home_country"))


class f_user_homeCity(Transformer):

    @keyword_only
    def __init__(self, **kwargs):
        super(f_user_homeCity, self).__init__()
        self.feature_names = "user_homeCity"

    def _transform(self, X):
        return X.withColumn(self.feature_names, col("home_city"))


class user_successfulReferrals_count(Transformer):

    @keyword_only
    def __init__(self, **kwargs):
        super(user_successfulReferrals_count, self).__init__()
        self.feature_names = "user_successfulReferrals_count"

    def _transform(self, X):
        return X.withColumn(self.feature_names, col("num_successful_referrals"))


class f_user_ageBucket(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_ageBucket, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_ageBucket"

    def _transform(self, X):
        @udf("string")
        def calc_ageBucket(number):
            if number < 21:
                return "a_20_and_younger"
            if 20 < number < 30:
                return "b_21_to_29"
            if 29 < number < 40:
                return "c_30_to_39"
            if 39 < number < 50:
                return "d_40_to_49"
            if 49 < number < 60:
                return "e_50_to_59"
            if number > 59:
                return "f_60_and_above"

        latest_date = self.transc_table.agg(
            f.max("created_date")).collect()[0][0]
        age = (X.withColumn("age", f.year(f.lit(latest_date).cast("date")) - col("birth_year"))
                .withColumn(self.feature_names, calc_ageBucket(col("age"))))
        return age.drop("age")


class f_user_daysSinceJoined(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_daysSinceJoined, self).__init__()
        self.transc_table = transc_table
        self.feature_names = "user_daysSinceJoined"

    def _transform(self, X):
        latest_date = self.transc_table.agg(
            f.max("created_date")).collect()[0][0]

        joined = (X.withColumn(self.feature_names,
                               f.datediff(f.lit(latest_date).cast("date"), col("joined_date").cast("date"))))
        return joined


class f_user_daysSinceLastTransc_quantiles(Transformer):

    @keyword_only
    def __init__(self, transc_table, **kwargs):
        super(f_user_daysSinceLastTransc_quantiles, self).__init__()
        self.transc_table = transc_table
        self.feature_names = ["user_daysBtwTransc_0q",
                              "user_daysBtwTransc_10q",
                              "user_daysBtwTransc_50q",
                              "user_daysBtwTransc_90q",
                              "user_daysBtwTransc_100q"]

    def _transform(self, X):
        calc_quantiles = f.expr(
            'percentile_approx(daysSinceLastTransc, array(0, 0.1, 0.5, 0.9, 1))')

        delta = (self.transc_table
                     .withColumn("prev_transc", f.lag("created_date", 1)
                                 .over(Window.partitionBy("user_id").orderBy("created_date")))
                     .withColumn("daysSinceLastTransc", f.datediff("created_date", col("prev_transc")))
                     .groupBy("user_id")
                     .agg(calc_quantiles.alias("q")))

        quantiles = (delta.select(["user_id"] +
                                  [delta.q[i].alias(colname) for i, colname
                                   in enumerate(self.feature_names)]))

        return (X.join(quantiles, ["user_id"], "left"))


class f_user_home_country(Transformer):

    @keyword_only
    def __init__(self, **kwargs):
        super(f_user_home_country, self).__init__()
        self.feature_names = "user_home_country"

    def _transform(self, X):
        return X.withColumn(self.feature_names, col("home_country"))


class f_user_home_city(Transformer):

    @keyword_only
    def __init__(self, **kwargs):
        super(f_user_home_city, self).__init__()
        self.feature_names = "user_home_city"

    def _transform(self, X):
        return X.withColumn(self.feature_names, col("home_city"))
