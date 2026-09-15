class OnesModel:
    def __init__(self):
        pass

    def fit(self, df):
        return self

    def predict(self, df):
        return [1] * len(df)
