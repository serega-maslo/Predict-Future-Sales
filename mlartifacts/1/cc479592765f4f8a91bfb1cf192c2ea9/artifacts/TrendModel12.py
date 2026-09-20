class TrendModel:
    def __init__(self, seasonality=12):
        self.df = None
        self.month = None
        self.seasonality = seasonality
        self.trend_df = None

    def fit(self, df):
        self.month = df['date_block_num'].max()
        columns_to_group = ['date_block_num', 'corrected_shop_id', 'corrected_item_id']
        df = (
            df.groupby(columns_to_group, as_index=False)
            .agg(item_cnt_day=("item_cnt_day", "sum"))
            .sort_values(columns_to_group)
        )

        df['trend'] = (
            df.groupby(['corrected_shop_id', 'corrected_item_id'])['item_cnt_day']
            .rolling(self.seasonality, center=True, min_periods=1)
            .mean()
            .values
        )

        self.trend_df = (
            df.groupby(
                ['date_block_num', 'corrected_shop_id', 'corrected_item_id'],
                as_index=False
            )
            .agg(trend_component=('trend', 'mean'))
        )

        self.df = (
            df[df['date_block_num'] == self.month]
            [['corrected_shop_id', 'corrected_item_id', 'item_cnt_day']]
        )

    def predict(self, df):
        X = df.copy()
        X = X.merge(
            self.df,
            on=["corrected_shop_id", "corrected_item_id"],
            how='left',
        )

        trend_last = self.month
        trend_prev = self.month - 1

        X = X.merge(
            self.trend_df[self.trend_df['date_block_num'] == trend_last]
            [['corrected_shop_id', 'corrected_item_id', 'trend_component']],
            on=['corrected_shop_id', 'corrected_item_id'],
            how='left'
        ).rename(columns={'trend_component': 'trend_last'})

        X = X.merge(
            self.trend_df[self.trend_df['date_block_num'] == trend_prev]
            [['corrected_shop_id', 'corrected_item_id', 'trend_component']],
            on=['corrected_shop_id', 'corrected_item_id'],
            how='left'
        ).rename(columns={'trend_component': 'trend_prev'})

        X['item_cnt_day'] = X['item_cnt_day'].fillna(0)
        X['trend_last'] = X['trend_last'].fillna(0)
        X['trend_prev'] = X['trend_prev'].fillna(X['trend_last'])

        X['trend_next'] = X['trend_last'] + (X['trend_last'] - X['trend_prev'])

        return (X['item_cnt_day'] - X['trend_last'] + X['trend_next']).clip(lower=0)
