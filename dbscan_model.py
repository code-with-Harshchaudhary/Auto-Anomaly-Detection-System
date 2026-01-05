from sklearn.cluster import DBSCAN

def detect_anomalies_dbscan(data):
    model = DBSCAN(eps=1.5, min_samples=3)
    return model.fit_predict(data)
