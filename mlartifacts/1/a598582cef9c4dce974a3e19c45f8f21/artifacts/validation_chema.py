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
