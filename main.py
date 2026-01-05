from data_loader import load_data
from preprocessing import scale_data
from isolation_forest import detect_anomalies_isolation_forest
from dbscan_model import detect_anomalies_dbscan
from autoencoder import train_autoencoder
import numpy as np

data = load_data("../data/sample_data.csv")
scaled_data = scale_data(data)

print("Isolation Forest:", detect_anomalies_isolation_forest(scaled_data))
print("DBSCAN:", detect_anomalies_dbscan(scaled_data))

autoencoder = train_autoencoder(scaled_data)
recon = autoencoder.predict(scaled_data)
error = np.mean((scaled_data - recon) ** 2, axis=1)
print("Autoencoder Reconstruction Error:", error)

