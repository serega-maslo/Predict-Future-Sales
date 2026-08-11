import numpy as np
import pandas as pd

class BaseLineModel:
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


class SeasonalModel:
    def __init__(self, seasonality=12):
        self.df = None
        self.month = None
        self.seasonality = seasonality

    def fit(self, df):
        self.month = df['date_block_num'].max()
        columns_to_group = ['date_block_num', 'corrected_shop_id', 'corrected_item_id']
        df = (
            df.groupby(columns_to_group, as_index=False)
            .agg(item_cnt_day=("item_cnt_day", "sum"))
        ).sort_values(columns_to_group)
        df['trend'] = (
            df.groupby(['corrected_shop_id', 'corrected_item_id'])['item_cnt_day']
            .rolling(self.seasonality, center=True, min_periods=1)
            .mean()
            .values
        )
        df['risidual'] = (
            df['item_cnt_day'] - df['trend']
        )

        df['season'] = (
            df['date_block_num'] % self.seasonality
        )

        self.seasonality_df = (
            df.groupby(
                ['season', 'corrected_shop_id', 'corrected_item_id'],
                as_index=False
            )
            .agg(
                seasonal_component=('risidual', 'mean')
            )
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
        season_last = self.month % self.seasonality
        season_next = (self.month + 1) % self.seasonality

        X = X.merge(
            self.seasonality_df[self.seasonality_df['season'] == season_last]
            [['corrected_shop_id', 'corrected_item_id', 'seasonal_component']],
            on=['corrected_shop_id', 'corrected_item_id'],
            how='left'
        ).rename(columns={'seasonal_component': 'season_last'})

        X = X.merge(
            self.seasonality_df[self.seasonality_df['season'] == season_next]
            [['corrected_shop_id', 'corrected_item_id', 'seasonal_component']],
            on=['corrected_shop_id', 'corrected_item_id'],
            how='left'
        ).rename(columns={'seasonal_component': 'season_next'})

        X['item_cnt_day'] = (X['item_cnt_day'].fillna(0))
        X['season_last'] = X['season_last'].fillna(0)
        X['season_next'] = X['season_next'].fillna(0)

        return (X['item_cnt_day'] - X['season_last'] + X['season_next']).clip(lower=0)



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


class SeasonalTrendModel:
    def __init__(self, seasonality=12):
        self.df = None
        self.month = None
        self.seasonality = seasonality
        self.seasonality_df = None
        self.trend_df = None

    def fit(self, df):
        self.month = df['date_block_num'].max()

        columns_to_group = [
            'date_block_num',
            'corrected_shop_id',
            'corrected_item_id'
        ]

        df = (
            df.groupby(columns_to_group, as_index=False)
            .agg(item_cnt_day=("item_cnt_day", "sum"))
            .sort_values(columns_to_group)
        )

        df['trend'] = (
            df.groupby(
                ['corrected_shop_id', 'corrected_item_id']
            )['item_cnt_day']
            .rolling(self.seasonality, center=True, min_periods=1)
            .mean()
            .values
        )

        df['risidual'] = df['item_cnt_day'] - df['trend']
        df['season'] = df['date_block_num'] % self.seasonality

        self.seasonality_df = (
            df.groupby(
                ['season', 'corrected_shop_id', 'corrected_item_id'],
                as_index=False
            )
            .agg(seasonal_component=('risidual', 'mean'))
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
            how='left'
        )

        season_last = self.month % self.seasonality
        season_next = (self.month + 1) % self.seasonality

        X = X.merge(
            self.seasonality_df[
                self.seasonality_df['season'] == season_last
            ][
                ['corrected_shop_id', 'corrected_item_id', 'seasonal_component']
            ],
            on=['corrected_shop_id', 'corrected_item_id'],
            how='left'
        ).rename(columns={'seasonal_component': 'season_last'})

        X = X.merge(
            self.seasonality_df[
                self.seasonality_df['season'] == season_next
            ][
                ['corrected_shop_id', 'corrected_item_id', 'seasonal_component']
            ],
            on=['corrected_shop_id', 'corrected_item_id'],
            how='left'
        ).rename(columns={'seasonal_component': 'season_next'})

        X = X.merge(
            self.trend_df[
                self.trend_df['date_block_num'] == self.month
            ][
                ['corrected_shop_id', 'corrected_item_id', 'trend_component']
            ],
            on=['corrected_shop_id', 'corrected_item_id'],
            how='left'
        ).rename(columns={'trend_component': 'trend_last'})

        X = X.merge(
            self.trend_df[
                self.trend_df['date_block_num'] == self.month - 1
            ][
                ['corrected_shop_id', 'corrected_item_id', 'trend_component']
            ],
            on=['corrected_shop_id', 'corrected_item_id'],
            how='left'
        ).rename(columns={'trend_component': 'trend_prev'})

        X['item_cnt_day'] = X['item_cnt_day'].fillna(0)
        X['season_last'] = X['season_last'].fillna(0)
        X['season_next'] = X['season_next'].fillna(0)
        X['trend_last'] = X['trend_last'].fillna(0)
        X['trend_prev'] = X['trend_prev'].fillna(X['trend_last'])

        X['trend_next'] = (
            X['trend_last'] +
            (X['trend_last'] - X['trend_prev'])
        )

        return (
            X['item_cnt_day']
            - X['trend_last']
            - X['season_last']
            + X['trend_next']
            + X['season_next']
        ).clip(lower=0)

