from pathlib import Path

import pandas as pd

from src.data.constants import INFLATION_DATA

def _compute_df_final(shops: pd.DataFrame, items: pd.DataFrame, item_cats: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:


    df_final = (
    train
    .merge(
        items[
            ["item_id", "corrected_item_id", "item_category_id", "item_name"]
        ],
        on="item_id",
        how="left"
    )
    .merge(
        item_cats[
            ["item_category_id", "main_category"]
        ],
        on="item_category_id",
        how="left"
    )
    .merge(
        shops[
            ["shop_id", "corrected_shop_id", "city"]
        ],
        on="shop_id",
        how="left"
    )
    .drop(columns=["item_category_id", "shop_id", "item_id"])
    )
    df_final.loc[484683, ["item_price", "no_inflation_price"]] = [1249.0, 1249.0 / INFLATION_DATA[2013, 5]]
    df_final.loc[1256153, ['item_price', 'item_cnt_day', 'no_inflation_price']] = [307980.0 / 522, 522, 289183.0985915493 / 522]
    indices_to_drop = df_final[df_final["item_name"] == "Доставка (EMS)"].nlargest(1, "item_price").index
    df_final = df_final.drop(indices_to_drop)
    df_final.drop([2909818, 2864235, 2851091, 2608040, 2626181, 2851073, 2067669, 2864260], inplace=True)
    indices_to_drop = df_final[df_final["item_name"] == "Доставка до пункта выдачи (Boxberry)"].nlargest(2, "item_cnt_day").index
    df_final = df_final.drop(indices_to_drop)
    return df_final
