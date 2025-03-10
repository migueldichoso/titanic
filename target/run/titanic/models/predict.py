
  
    
    
import pandas as pd
from sklearn.impute import SimpleImputer
from snowflake.ml.registry import registry


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
  df = df[["PCLASS", "SEX", "AGE", "SIBSP", "PARCH", "FARE", "EMBARKED"]]
  df["PCLASS"] = pd.Categorical(df["PCLASS"], categories=[1, 2, 3])
  df["SEX"] = pd.Categorical(df["SEX"], categories=["male", "female"])
  df["EMBARKED"] = pd.Categorical(df["EMBARKED"], categories=["C", "Q", "S"])
  return pd.get_dummies(df, columns=["PCLASS", "SEX", "EMBARKED"])


def model(dbt, session):
  dbt.config(
    materialized="table",
    python_version="3.11",
    packages=["snowflake-ml-python", "pandas", "scikit-learn"],
  )

  dataset = dbt.ref("titanic3")

  data = dataset.to_pandas()

  x = preprocess(data)

  imputer = SimpleImputer()
  x = imputer.fit_transform(x)

  reg = registry.Registry(session=session)

  model_ref = dbt.ref("train_model")
  mv = reg.get_model(model_ref.table_name).default
  data["PREDICTED"] = mv.run(x, function_name="PREDICT")

  return data


# This part is user provided model code
# you will need to copy the next section to run the code
# COMMAND ----------
# this part is dbt logic for get ref work, do not modify

def ref(*args, **kwargs):
    refs = {"titanic3": "analytics.aaa_titanic_miguel.titanic3", "train_model": "analytics.aaa_titanic_miguel.train_model"}
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
    identifier = "predict"
    
    def __repr__(self):
        return 'analytics.aaa_titanic_miguel.predict'


class dbtObj:
    def __init__(self, load_df_function) -> None:
        self.source = lambda *args: source(*args, dbt_load_df_function=load_df_function)
        self.ref = lambda *args, **kwargs: ref(*args, **kwargs, dbt_load_df_function=load_df_function)
        self.config = config
        self.this = this()
        self.is_incremental = False

# COMMAND ----------



def materialize(session, df, target_relation):
    # make sure pandas exists
    import importlib.util
    package_name = 'pandas'
    if importlib.util.find_spec(package_name):
        import pandas
        if isinstance(df, pandas.core.frame.DataFrame):
          session.use_database(target_relation.database)
          session.use_schema(target_relation.schema)
          # session.write_pandas does not have overwrite function
          df = session.createDataFrame(df)
    
    df.write.mode("overwrite").save_as_table('analytics.aaa_titanic_miguel.predict', table_type='transient')

def main(session):
    dbt = dbtObj(session.table)
    df = model(dbt, session)
    materialize(session, df, dbt.this)
    return "OK"

  