from typing import Protocol
from pathlib import Path
import inspect

import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import root_mean_squared_error
import shap


class ModelProtocol(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelProtocol": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def validation_chema(df: pd.DataFrame, y: pd.DataFrame, model_class: ModelProtocol, features: list[str], model_params: dict, start: int = 0, window_size: int = -1, importance: bool = False, encode: bool = False)->pd.DataFrame:
    df['item_cnt_day'] = y
    df['date'] = pd.to_datetime(df['date'])
    results = []
    feature_importances_list = []
    max_month = df['date_block_num'].max()
    if window_size != -1:
        max_month -= window_size - 1
    columns_to_group = ['corrected_shop_id', 'corrected_item_id', 'main_category']
    train = df.iloc[0:0].copy()
    valid_features = list(set(features) | {'date_block_num', 'date', 'corrected_shop_id', 'corrected_item_id'})
    for month in range(start, max_month):
        if window_size == -1:
            train_idx = df["date_block_num"] <= month
            val_month = month + 1
        else:
            train_idx = (df['date_block_num'] >= month) & (df['date_block_num'] < month + window_size)
            val_month = month + window_size
        train = df.loc[train_idx, features + ['item_cnt_day']].copy()
        valid = df[df['date_block_num'] == val_month].copy()
        valid['item_cnt_day'] = 0



        if encode:

            train_indices = train.index.copy()
            
            train = pd.concat((train, valid[train.columns]), ignore_index=True)

            category_sales = train.groupby(["main_category"], as_index=False).agg(item_cnt_day=("item_cnt_day", "sum")).sort_values(by=["item_cnt_day"], ascending=False).reset_index(drop=True)
            category_map = pd.Series(category_sales.index.values, index=category_sales["main_category"])
            train["main_category"] = train["main_category"].map(category_map)
            valid["main_category"] = valid["main_category"].map(category_map)


            city_sales = train.groupby(["city"], as_index=False).agg(item_cnt_day=("item_cnt_day", "sum")).sort_values(by=["item_cnt_day"], ascending=False).reset_index(drop=True)
            city_map = pd.Series(city_sales.index.values, index=city_sales["city"])
            train["city"] = train["city"].map(city_map)
            valid["city"] = valid["city"].map(city_map)


            
            item_sales = train.groupby(["corrected_item_id"], as_index=False).agg(item_cnt_day=("item_cnt_day", "sum")).sort_values(by=["item_cnt_day"], ascending=False).reset_index(drop=True)
            item_map = pd.Series(item_sales.index.values, index=item_sales["corrected_item_id"])
            train["corrected_item_id"] = train["corrected_item_id"].map(item_map)
            valid["corrected_item_id"] = valid["corrected_item_id"].map(item_map)


            shop_sales = train.groupby(["corrected_shop_id"], as_index=False).agg(item_cnt_day=("item_cnt_day", "sum")).sort_values(by=["item_cnt_day"], ascending=False).reset_index(drop=True)
            shop_map = pd.Series(shop_sales.index.values, index=shop_sales["corrected_shop_id"])
            train["corrected_shop_id"] = train["corrected_shop_id"].map(shop_map)
            valid["corrected_shop_id"] = valid["corrected_shop_id"].map(shop_map)

            train = train.iloc[:len(train_indices)].copy()
            train.index = train_indices


        train = train[train['date_block_num'] < val_month]
        model = model_class(**model_params)
        model.fit(train[features], y.loc[train.index])

        if importance:
            fold_importance = pd.Series(model.feature_importances_, index=features)
            feature_importances_list.append(fold_importance)
        
        # explainer = shap.TreeExplainer(model)
        # shap_values = explainer(valid[features])

        # fold_shap = pd.DataFrame(
        #     shap_values.values,
        #     columns=features,
        #     index=valid.index,
        # )

        # feature_importances_list.append(fold_shap.abs().mean())


        prediction = model.predict(valid[features])
        valid['item_cnt_day'] = prediction
        valid['date'] = df.loc[df.index[df['date_block_num'] == val_month]]['date']
        if window_size != -1:
            valid['true_ans'] = y.loc[df.index[df['date_block_num'] == val_month]].values
        else:
            valid['true_ans'] = y.loc[df.index[df['date_block_num'] == val_month]].values
        #results = pd.concat([results, valid[['date_block_num', 'date', 'corrected_shop_id', 'corrected_item_id', 'item_cnt_day', 'true_ans']]], ignore_index=True)
        results.append(valid[['date_block_num', 'date', 'corrected_shop_id', 'corrected_item_id', 'item_cnt_day', 'true_ans']])
    return pd.concat(results, ignore_index=True), pd.DataFrame(feature_importances_list)


def log_validation(df: pd.DataFrame, y:pd.DataFrame, model_class: ModelProtocol, features: list[str], save_path: Path, model_params: dict | None = None, model_name: str | None = None, start: int = 0, window_size: int = -1, importance: bool = False, encode: bool=False):
    if model_name is None:
        model_name = model_class.__name__

    if model_params is None:
        model_params = {}

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model_class", model_name)
        mlflow.log_param("start", start)
        mlflow.log_param("window_size", window_size)

        mlflow.log_params(model_params)

        results, importance  = validation_chema(df=df, y=y, model_class=model_class, features=features, model_params=model_params, start=start, window_size=window_size, importance=importance, encode=encode)

        #y_true = results["true_ans"].astype(float)
        #y_pred = results["item_cnt_day"].astype(float)

        rmse = results.groupby("date_block_num").apply(
            lambda x: root_mean_squared_error(x["true_ans"], x["item_cnt_day"])
        ).mean()

        

        mlflow.log_metric("rmse", rmse)

        predictions_path = save_path / f"validation_predictions_{model_name}.parquet"
        results.to_parquet(predictions_path, index=False)
        mlflow.log_artifact(predictions_path)

        importance_path = save_path / f"validation_importance_{model_name}.parquet"
        importance.to_parquet(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        validation_shema_path = save_path / "validation_chema.py"
        validation_shema_path.write_text(
            inspect.getsource(validation_chema),
            encoding="utf-8"
        )
        mlflow.log_artifact(validation_shema_path)


        model_code_path = save_path / f"{model_name}.py"
        model_code_path.write_text(
            inspect.getsource(model_class),
            encoding="utf-8"
        )
        mlflow.log_artifact(model_code_path)

        return results, importance