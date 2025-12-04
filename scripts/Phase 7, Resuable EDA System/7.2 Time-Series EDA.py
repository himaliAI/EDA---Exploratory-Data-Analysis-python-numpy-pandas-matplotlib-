import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['diagnosis'] = df['target'].map({0:"Malignant", 1:"Benign"})

# Rolling moving mean/average
df['rolling_mean_7'] = df['mean radius'].rolling(window=7).mean()

# rolling Variance / Std Dev
df['rolling_std_7'] = df['mean radius'].rolling(window=7).std()
    # print(df[['mean radius', 'rolling_mean_7', 'rolling_std_7']].tail(20).round(2))

# helper functions for time-series EDA
def rolling_summary(series, window=7):
    # returns mean and SD
    return pd.DataFrame({
        "Rolling Mean": series.rolling(window).mean().round(2),
        "Rolling Std": series.rolling(window).std().round(2)
    })

def seasonal_report(series, period=12, model='additive'):
    # performs seasonal decomposition and returns components
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(series, model=model, period=period)
    return result
