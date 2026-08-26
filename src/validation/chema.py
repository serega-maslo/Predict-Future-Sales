from typing import Protocol
from pathlib import Path
import inspect

import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import root_mean_squared_error


class ModelProtocol(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelProtocol": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def validation_chema(df: pd.DataFrame, model_class: ModelProtocol, start: int = 0, window_size: int = -1)->pd.DataFrame:
    results = df[['date_block_num', 'corrected_shop_id', 'corrected_item_id', 'item_cnt_day']].iloc[0:0].copy()
    results['true_ans'] = None
    max_month = df['date_block_num'].max()
    if window_size != -1:
        max_month -= window_size - 1
    columns_to_group = ['corrected_shop_id', 'corrected_item_id']
    train = df.iloc[0:0].copy()
    for month in range(start, max_month):
        if window_size == -1:
            train = pd.concat(
                [train, df[df["date_block_num"] == month]],
                ignore_index=True,
            )
        else:
            train = df[(df['date_block_num'] >= month) & (df['date_block_num'] < month + window_size)]
        
        valid = df[df["date_block_num"] == month + 1][['corrected_shop_id', 'corrected_item_id', 'item_cnt_day']]
        valid = (
            valid
            .groupby(columns_to_group, as_index=False)
            .agg(item_cnt_day=("item_cnt_day", "sum"))
        )
        valid = valid.rename(columns={'item_cnt_day': 'true_ans'})
        valid['date_block_num'] = month + 1
        model = model_class()
        model.fit(train)
        prediction = model.predict(valid)
        valid['item_cnt_day'] = prediction
        results = pd.concat([results, valid], ignore_index=True)
    return results


def log_validation(df: pd.DataFrame, model_class, save_path: Path, model_name: str | None = None, start: int = 0, window_size: int = -1):
    if model_name is None:
        model_name = model_class.__name__

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model_class", model_name)
        mlflow.log_param("start", start)
        mlflow.log_param("window_size", window_size)

        results = validation_chema(df=df, model_class=model_class, start=start, window_size=window_size)

        y_true = results["true_ans"].astype(float)
        y_pred = results["item_cnt_day"].astype(float)

        mlflow.log_metric("rmse", root_mean_squared_error(y_true, y_pred))

        predictions_path = save_path / f"validation_predictions_{model_name}.parquet"
        results.to_parquet(predictions_path, index=False)
        mlflow.log_artifact(predictions_path)

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

        return results