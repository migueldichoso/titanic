
  create or replace   view analytics.aaa_titanic_miguel.train
  
   as (
    import datetime
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from snowflake.ml.model import model_signature


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
  df = df[["employee_id","age","monthly_expenses"]]
  return pd.get_dummies(df, columns=["employee_id", "age", "monthly_expenses"])


def model(dbt, session):
  dbt.config(
    materialized="model",
    python_version="3.11",
    packages=["snowflake-ml-python", "pandas", "scikit-learn"],
    schema="models",
  )

  dataset = dbt.ref("train_data")

  data = dataset.to_pandas()

  x = preprocess(data)
  y = data["attrition"]

  imputer = SimpleImputer()
  x = imputer.fit_transform(x)

  model = SVC()
  model.fit(x, y)

  return {
    "model": model,
    "signatures": {"predict": model_signature.infer_signature(x, y)},
    "version_name": datetime.datetime.today().strftime("V%Y%m%d"),
    "metrics": {"r2_score": model.score(x, y)},
    "comment": f"r2_score: {model.score(x, y)}",
    "set_default": True,
  }
  );

