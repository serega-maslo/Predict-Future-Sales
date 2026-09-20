from pathlib import Path

import pandas as pd

def _compute_for_submission(raw_data_path: Path, shops: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    for_submission = pd.read_csv(raw_data_path / "test.csv")
    for_submission = (
    for_submission
    .merge(
        items[
            ["item_id", "corrected_item_id"]
        ],
        on="item_id",
        how="left"
    )
    .merge(
        shops[
            ["shop_id", "corrected_shop_id"]
        ],
        on="shop_id",
        how="left"
    )
    )
    return for_submission
