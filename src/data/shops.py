from pathlib import Path

import pandas as pd

from src.data.compute_utils import _clean_shop_name, _get_shop_city


def _compute_shops(raw_data_path: Path) -> pd.DataFrame:
    shops = pd.read_csv(raw_data_path / "shops.csv")
    shops['shop_name'] = shops["shop_name"].apply(_clean_shop_name)
    shops['city'] = shops['shop_name'].apply(_get_shop_city)
    shops['corrected_shop_id'] = shops['shop_id']
    shops.loc[shops['shop_name'] == 'Якутск Орджоникидзе, 56 фран', ['corrected_shop_id']] = [shops[shops["shop_name"] == 'Якутск Орджоникидзе, 56'].iloc[0]['shop_id']]
    shops.loc[shops['shop_name'] == 'Якутск ТЦ "Центральный" фран', ['corrected_shop_id']] = [shops[shops["shop_name"] == 'Якутск ТЦ "Центральный"'].iloc[0]['shop_id']]
    shops.loc[shops['shop_name'] == 'Жуковский ул. Чкалова 39м?', ['corrected_shop_id']] = [shops[shops["shop_name"] == 'Жуковский ул. Чкалова 39м²'].iloc[0]['shop_id']]
    shops['is_fran'] = 0
    shops[shops["shop_name"] == 'Якутск ТЦ "Центральный" фран'].iloc[0]['is_fran'] = 1
    shops[shops["shop_name"] == 'Якутск Орджоникидзе, 56 фран'].iloc[0]['is_fran'] = 1
    return shops
