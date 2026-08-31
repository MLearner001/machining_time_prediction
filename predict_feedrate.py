import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import os

# Hyperparameters must match the training script exactly
LOOK_BACK = 50
LOOK_AHEAD = 50

def predict_machining_time(test_file_path, model_path="bi_lstm_lookahead_model.keras"):
    print(f"[+] Memuat model dari {model_path}...")
    try:
        # custom_objects is required to bypass the Keras 3 MSE bug
        model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    except Exception as e:
        print(f"[!] Gagal memuat model. Pastikan file ada. Error: {e}")
        return

    print("[+] Memuat scaler (x_scaler.pkl dan y_scaler.pkl)...")
    try:
        x_scaler = joblib.load("x_scaler.pkl")
        y_scaler = joblib.load("y_scaler.pkl")
    except Exception as e:
        print(f"[!] Gagal memuat scaler. Pastikan Anda sudah melatih model dengan script terbaru yang menyimpan scaler. Error: {e}")
        return

    print(f"[+] Membaca file tes: {test_file_path}")
    df = pd.read_csv(test_file_path)

    if 'Actual_Feedrate' not in df.columns or 'Time(s)' not in df.columns:
        print("[!] File CSV harus memiliki kolom 'Actual_Feedrate' dan 'Time(s)'.")
        return

    # Extract raw data
    X_raw = df.drop(columns=['Time(s)', 'Actual_Feedrate'])
    actual_feedrate = df['Actual_Feedrate'].values
    actual_time = df['Time(s)'].values

    print("[+] Menyiapkan sliding windows (LOOK_BACK=50, LOOK_AHEAD=50)...")
    # Scale input data
    X_scaled = x_scaler.transform(X_raw).astype(np.float32)
    num_features = X_scaled.shape[1]

    # Add padding to the ends so we can predict the very first and very last rows
    past_padding = np.zeros((LOOK_BACK, num_features), dtype=np.float32)
    future_padding = np.zeros((LOOK_AHEAD, num_features), dtype=np.float32)
    X_padded = np.vstack((past_padding, X_scaled, future_padding))

    total_rows = len(X_scaled)

    print("[+] Melakukan Prediksi Digital Twin menggunakan GPU...")
    # Predict using Generator to strictly prevent RAM OOM ballooning on huge files
    def data_generator():
        for i in range(len(X_scaled)):
            center = i + LOOK_BACK
            yield X_padded[center - LOOK_BACK : center + LOOK_AHEAD + 1]

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=tf.TensorSpec(shape=(LOOK_BACK + 1 + LOOK_AHEAD, num_features), dtype=tf.float32)
    ).batch(1024).prefetch(tf.data.AUTOTUNE)

    # Predict scaled feedrates
    predicted_scaled = model.predict(dataset, verbose=1)

    # Reverse the scaling to get real mm/min values!
    predicted_feedrate = y_scaler.inverse_transform(predicted_scaled).flatten()

    # Prevent negative feedrates (impossible in reality)
    predicted_feedrate = np.maximum(predicted_feedrate, 0.0)

    print("\n[+] ===========================================")
    print("[+] KALKULASI WAKTU MACHINING")

    # Simple time calculation based on log differences
    # Since logs are roughly 4ms apart, we calculate expected time if traveling at predicted speeds vs actual
    # Note: For strict geometric distance calculation, we would need X,Y,Z columns.
    # Here we calculate average error metrics.

    mae = np.mean(np.abs(predicted_feedrate - actual_feedrate))
    mape = np.mean(np.abs((actual_feedrate - predicted_feedrate) / np.maximum(actual_feedrate, 1e-6))) * 100

    print(f"    - Rata-rata Error (MAE): {mae:.2f} mm/min")
    print(f"    - Persentase Error (MAPE): {mape:.2f} %")
    print("[+] ===========================================\n")

    print("[+] Menyimpan grafik perbandingan...")
    plt.figure(figsize=(15, 6))

    # Plotting only the first 5000 rows to make the graph readable, otherwise it's just a solid block of ink
    limit = min(5000, total_rows)
    plt.plot(actual_time[:limit], actual_feedrate[:limit], label="Actual Feedrate (SinuTrain)", color="blue", alpha=0.7)
    plt.plot(actual_time[:limit], predicted_feedrate[:limit], label="Predicted Feedrate (Digital Twin)", color="red", alpha=0.7, linestyle="dashed")

    plt.title(f"Digital Twin Feedrate Prediction (First {limit} rows)", fontsize=14, fontweight='bold')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Feedrate (mm/min)")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig("prediction_comparison.png", dpi=300)
    print("[✔] Selesai! Grafik disimpan sebagai 'prediction_comparison.png'.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Penggunaan: python predict_feedrate.py <path_to_test_csv> [path_to_model]")
    else:
        test_file = sys.argv[1]
        model_file = sys.argv[2] if len(sys.argv) > 2 else "bi_lstm_lookahead_model.keras"
        predict_machining_time(test_file, model_file)
