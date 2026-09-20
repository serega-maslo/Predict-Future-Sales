class ZerosModel:
    def __init__(self):
        pass

    def fit(self, df):
        return self

    def predict(self, df):
        return [0] * len(df)
