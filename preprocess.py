import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[['Close']])

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(data['Target'].iloc[i])

    return np.array(X), np.array(y), scaler