import re

import numpy as np
import pandas as pd

def _get_main_category(item_category_name):
    return item_category_name.split('-', 1)[0].strip()

def _clean_item_name(text):
        text = re.sub(r'^[\!\*\/\s]+', '', text)
        text = re.sub(r'[\!\*\/\s]$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+D$', ' Disc', text, flags=re.IGNORECASE)
        return text.strip()


def _clean_shop_name(text):
    return text.lstrip('!')


def _get_shop_city(text):
    city = text.split()[0]
    if city == 'Выездная':
        city = 'Выездная торговля'
    elif city == 'Цифровой' or city == 'Интернет-магазин':
        city = 'Интернет'
    return city
