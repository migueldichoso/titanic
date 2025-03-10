import datetime
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from snowflake.ml.model import model_signature


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
  df = df[["CUSTOMER_ID", "AGE", "MONTHLY_EXPENSES"]]
  df["CUSTOMER_ID"] = pd.Categorical(df["CUSTOMER_ID"])
  df["AGE"] = pd.Categorical(df["AGE"])
  df["MONTHLY_EXPENSES"] = pd.Categorical(df["MONTHLY_EXPENSES"])
  return pd.get_dummies(df, columns=["CUSTOMER_ID", "AGE", "MONTHLY_EXPENSES"])


def model(dbt, session):
  dbt.config(
    materialized="model",
    python_version="3.11",
    packages=["snowflake-ml-python", "pandas", "scikit-learn"],
  )

  dataset = dbt.ref("train_data")

  data = dataset.to_pandas()

  x = preprocess(data)
  y = data["ATTRITION"]

  imputer = SimpleImputer()
  x = imputer.fit_transform(x)

  model = SVC()
  model.fit(x, y)

  return {
    "model": model,
    "signatures": {"predict": model_signature.infer_signature(x, y)},
    "version_name": datetime.datetime.today().strftime("V%Y%m%d%s"),
    "metrics": {"r2_score": model.score(x, y)},
    "comment": f"r2_score: {model.score(x, y)}",
    "set_default": True,
  }


# This part is user provided model code
# you will need to copy the next section to run the code
# COMMAND ----------
# this part is dbt logic for get ref work, do not modify

def ref(*args, **kwargs):
    refs = {"train_data": "analytics.aaa_titanic_miguel.train_data"}
    key = '.'.join(args)
    version = kwargs.get("v") or kwargs.get("version")
    if version:
        key += f".v{version}"
    dbt_load_df_function = kwargs.get("dbt_load_df_function")
    return dbt_load_df_function(refs[key])


def source(*args, dbt_load_df_function):
    sources = {}
    key = '.'.join(args)
    return dbt_load_df_function(sources[key])


config_dict = {}


class config:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def get(key, default=None):
        return config_dict.get(key, default)

class this:
    """dbt.this() or dbt.this.identifier"""
    database = "analytics"
    schema = "aaa_titanic_miguel"
    identifier = "train"
    
    def __repr__(self):
        return 'analytics.aaa_titanic_miguel.train'


class dbtObj:
    def __init__(self, load_df_function) -> None:
        self.source = lambda *args: source(*args, dbt_load_df_function=load_df_function)
        self.ref = lambda *args, **kwargs: ref(*args, **kwargs, dbt_load_df_function=load_df_function)
        self.config = config
        self.this = this()
        self.is_incremental = False

# COMMAND ----------


