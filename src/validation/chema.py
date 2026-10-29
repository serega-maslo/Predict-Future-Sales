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


def validation_chema(df: pd.DataFrame, y: pd.DataFrame, model_class: ModelProtocol, features: list[str], model_params: dict, start: int = 0, window_size: int = -1, importance: bool = False)->pd.DataFrame:
    
    #results = df[['date_block_num', 'date', 'corrected_shop_id', 'corrected_item_id', 'item_cnt_day']].iloc[0:0].copy()
    #results['true_ans'] = None
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
        else:
            train_idx = (df['date_block_num'] >= month) & (df['date_block_num'] < month + window_size)
        train = df.loc[train_idx, features]
            
        
        #valid = df[df["date_block_num"] == month + 1][['date', 'corrected_shop_id', 'corrected_item_id', 'main_category',  'item_cnt_day']]
        #valid = df[df["date_block_num"] == month + 1][valid_features]
        if window_size != -1:
            valid = df[df["date_block_num"] == month + window_size][valid_features]
        else:
            valid = df[df["date_block_num"] == month + 1][valid_features]
        #valid['date_block_num'] = month + 1
        valid['date'] = pd.to_datetime(valid['date'])
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
        if window_size != -1:
            valid['true_ans'] = y.loc[df.index[df['date_block_num'] == month + window_size]].values
        else:
            valid['true_ans'] = y.loc[df.index[df['date_block_num'] == month + 1]].values
        #results = pd.concat([results, valid[['date_block_num', 'date', 'corrected_shop_id', 'corrected_item_id', 'item_cnt_day', 'true_ans']]], ignore_index=True)
        results.append(valid[['date_block_num', 'date', 'corrected_shop_id', 'corrected_item_id', 'item_cnt_day', 'true_ans']])
    return pd.concat(results, ignore_index=True), pd.DataFrame(feature_importances_list)


def log_validation(df: pd.DataFrame, y:pd.DataFrame, model_class: ModelProtocol, features: list[str], save_path: Path, model_params: dict | None = None, model_name: str | None = None, start: int = 0, window_size: int = -1, importance: bool = False):
    if model_name is None:
        model_name = model_class.__name__

    if model_params is None:
        model_params = {}

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model_class", model_name)
        mlflow.log_param("start", start)
        mlflow.log_param("window_size", window_size)

        mlflow.log_params(model_params)

        results, importance  = validation_chema(df=df, y=y, model_class=model_class, features=features, model_params=model_params, start=start, window_size=window_size, importance=importance)

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