import yaml
import logging
import mlflow
import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import lightgbm as lgb
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import yaml
import logging



with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
mlflow.set_experiment(cfg["mlflow_experiment"])

# Logging module for info-level messages thoughout the pipeline
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    """
    1) Reads in raw CSV, renames date → 'ds' and target → 'y'.
    2) Groups by 'ds' to sum all values in that month (removes duplicates).
    3) Resamples to a regular monthly (MS) index, forward-fills gaps.
    """
    df = pd.read_csv(path, parse_dates=[cfg["date_col"]])
    df = df.rename(columns={cfg["date_col"]: "ds", cfg["target_col"]: "y"})
    
    # 2) Aggregate duplicates so each 'ds' is unique
    df = df.groupby("ds", as_index=False)["y"].sum()
    
    # 3) Turn into a regular time series at month start frequency
    df = (
        df
        .set_index("ds")
        .resample("MS")      # sum within each month (will leave gaps for missing months)
        .sum()
        .fillna(method="ffill")
        .reset_index()
    )
    return df



def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["month"] = df["ds"].dt.month
    df["quarter"] = df["ds"].dt.quarter

    for lag in cfg["lags"]:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    df["rolling_3m"] = df["y"].rolling(3).mean()
    df["rolling_6m"] = df["y"].rolling(6).mean()
    return df.dropna()

def evaluate(true, pred, name):

    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))\
    
    return {f"{name}_MAE": mae, f"{name}_RMSE": rmse}

def train_prophet(train_df: pd.DataFrame) -> Prophet:
    """
    Builds and fits a Prophet model with yearly + quarterly seasonality.
    """
    m = Prophet(
        yearly_seasonality=cfg["prophet"]["yearly"],
        changepoint_prior_scale=cfg["prophet"]["cps"]
    )
    m.add_seasonality(
        name="quarterly",  # corrected spelling
        period=cfg["prophet"]["quarterly_period"],
        fourier_order=cfg["prophet"]["fourier_order"]
    )
    # also fix the typo in “columns” here:
    m.fit(train_df.rename(columns={"ds": "ds", "y": "y"}))
    return m


def train_sarimax(train_series):

    best_aic, best_cfg = np.inf, None
    p = d = q = cfg["sarimax"]["pdq_range"]
    seasonal = [(i,j,k,12) for i in p for j in d for k in q]
    for order in [(i,j,k) for i in p for j in d for k in q]:
        for s in seasonal:
            try:
                mod = SARIMAX(
                    train_series,
                    order=order,
                    seasonal_order=s,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                res = mod.fit(disp=False)
                if res.aic < best_aic:
                    best_aic, best_cfg = res.aic, (order, s)
            except:
                continue
    logger.info(f"Best SARIMAX: {best_cfg} → AIC={best_aic}")
    return SARIMAX(
        train_series,
        order=best_cfg[0],
        seasonal_order=best_cfg[1],
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)


def train_lgb(train_X: pd.DataFrame, train_y: pd.Series) -> lgb.LGBMRegressor:
    """
    Fit a LightGBM model with constrained complexity so it can split on small datasets.
    """
    model = lgb.LGBMRegressor(
        num_leaves=cfg["lgb"]["num_leaves"],
        min_data_in_leaf=cfg["lgb"]["min_data_in_leaf"],
        min_data_in_bin=cfg["lgb"]["min_data_in_bin"],
        n_estimators=cfg["lgb"]["n_estimators"],
        learning_rate=cfg["lgb"]["learning_rate"]
    )
    model.fit(train_X, train_y)
    return model

def explain_model(model, X: pd.DataFrame, y: pd.Series = None):
    """
    Uses LightGBM's built-in split-based feature importances.
    """
    imp_df = (
        pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    logger.info("\n" + imp_df.to_string(index=False))

    imp_df.to_csv(cfg["feature_imp_csv"], index=False)



def run_pipeline():
    # 6.1 Load & Prepare Data
    df    = load_data(cfg["data_path"])
    df_fe = feature_engineering(df)

    # 6.2 Train/Test Split
    horizon = cfg["forecast_horizon"]
    train, test = df_fe[:-horizon], df_fe[-horizon:]

    # 6.3 MLflow Experiment Tracking
    with mlflow.start_run(run_name="forecasting_run"):
        mlflow.log_params(cfg)

        # ── Prophet Forecasting ──
        prop   = train_prophet(train[["ds","y"]])
        future = prop.make_future_dataframe(periods=horizon, freq="MS")
        fc_prop = prop.predict(future).set_index("ds")["yhat"].loc[test["ds"]]
        mlflow.log_metric("prophet_MAE", evaluate(test["y"], fc_prop, "prophet")["prophet_MAE"])

        # ── SARIMAX Forecasting ──
        sar    = train_sarimax(train.set_index("ds")["y"])
        fc_sar = sar.get_forecast(steps=horizon).predicted_mean
        mlflow.log_metric("sarimax_MAE", evaluate(test["y"], fc_sar, "sarimax")["sarimax_MAE"])

          # ── LightGBM Forecast ──
        # ── Debug: check feature alignment & variance ──
        print("CFG features list:", cfg["features"])
        print("Train columns:", train.columns.tolist())
        for feat in cfg["features"]:
            if feat not in train.columns:
                print(f">>> FEATURE MISSING: '{feat}' not in train.columns")
        print("Head of feature DataFrame:")
        print(train[cfg["features"]].head())
        print("Variance of each feature:")
        print(train[cfg["features"]].var().to_dict())

        # ── LightGBM Forecast ──
        features = cfg["features"]
        lgbm     = train_lgb(train[features], train["y"])

        fc_lgb   = []
        history  = train.copy()

        for ds, true in zip(test["ds"], test["y"]):
            # 1) Predict on the very last row's features
            Xp   = history[features].iloc[[-1]]
            pred = lgbm.predict(Xp)[0]
            fc_lgb.append(pred)

            # 2) Build a flat dict from the last row, overwrite ds & y
            new_row = history.iloc[-1].to_dict()
            new_row["ds"] = ds
            new_row["y"]  = true

            # 3) Append via a one-row DataFrame built from the dict
            history = pd.concat(
                [history, pd.DataFrame([new_row])],
                ignore_index=True
            )

        fc_lgb = pd.Series(fc_lgb, index=test["ds"])
        mlflow.log_metric("lgbm_MAE",
                         evaluate(test["y"], fc_lgb, "lgbm")["lgbm_MAE"])

        # ── Ensemble & Metrics ──
        ensemble = pd.concat([fc_prop, fc_sar, fc_lgb], axis=1).mean(axis=1)
        metrics  = evaluate(test["y"], ensemble, "ensemble")
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # ── Explain & Save Model ──
        # ── Explain & Save Model ──
        explain_model(lgbm, train[features], train["y"])
        mlflow.sklearn.log_model(lgbm, "lgbm_model")


        # ── Export Final Forecast ──
        final_fc = pd.DataFrame({
            "ds": future["ds"].iloc[-horizon:],
            "yhat_prophet": prop.predict(future)["yhat"].iloc[-horizon:],
            "yhat_sarimax": sar.get_forecast(horizon).predicted_mean.values,
            "yhat_lgbm": ensemble.values
        })
        final_fc.to_csv(cfg["output_csv"], index=False)
        mlflow.log_artifact(cfg["output_csv"])

if __name__ == "__main__":
    run_pipeline()