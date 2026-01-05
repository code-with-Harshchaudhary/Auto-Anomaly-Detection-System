from sklearn.ensemble import IsolationForest

def detect_anomalies_isolation_forest(data):
    model = IsolationForest(contamination=0.1, random_state=42)
    return model.fit_predict(data)
