from sklearn.preprocessing import StandardScaler

def scale_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)
