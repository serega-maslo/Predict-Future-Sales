from pathlib import Path

import numpy as np
import pandas as pd

from src.data.compute_utils import _get_main_category


def _compute_item_cats(raw_data_path: Path) -> pd.DataFrame:
    item_cats = pd.read_csv(raw_data_path / "item_categories.csv")
    item_cats["main_category"] = item_cats["item_category_name"].map(_get_main_category)
    return item_cats