import pandas as pd

class BaseLineModel():
    def __init__(self):
        self.df = None
        self.month = None


    def fit(self, df):
        self.month = df['date_block_num'].max()
        columns_to_group = ['date_block_num', 'corrected_shop_id', 'corrected_item_id']
        self.df = (
            df[df['date_block_num'] == self.month]
            .groupby(columns_to_group, as_index=False)
            .agg(item_cnt_day=("item_cnt_day", "sum"))
        )


    def predict(self, df):
        X = df.copy()
        X = X.merge(
            self.df,
            on=["corrected_shop_id", "corrected_item_id"],
            how='left',
        )
        X['item_cnt_day'] = (X['item_cnt_day'].fillna(0))
        return X['item_cnt_day']
