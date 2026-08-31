import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import sys

# Hyperparameters
LOOK_BACK = 50
LOOK_AHEAD = 50
CYCLE_TIME = 0.004  # 4ms SinuTrain log frequency

def evaluate_directory(test_dir, model_path="bi_lstm_lookahead_model.keras"):
    if not os.path.exists(test_dir):
        print(f"[!] Direktori {test_dir} tidak ditemukan.")
        return

    print(f"[+] Memuat model dari {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    except Exception as e:
        print(f"[!] Gagal memuat model. Error: {e}")
        return

    print("[+] Memuat scaler (x_scaler.pkl dan y_scaler.pkl)...")
    try:
        x_scaler = joblib.load("x_scaler.pkl")
        y_scaler = joblib.load("y_scaler.pkl")
    except Exception as e:
        print(f"[!] Gagal memuat scaler. Error: {e}")
        return

    csv_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"[!] Tidak ada file CSV di {test_dir}")
        return

    print(f"\n[+] Ditemukan {len(csv_files)} file. Memulai evaluasi batch...\n")
    print(f"{'File Name':<35} | {'Actual Time (s)':<15} | {'Predicted Time (s)':<18} | {'Accuracy (%)':<15}")
    print("-" * 90)

    total_actual_time_all = 0.0
    total_predicted_time_all = 0.0

    for file_name in sorted(csv_files):
        file_path = os.path.join(test_dir, file_name)
        df = pd.read_csv(file_path)

        if 'Actual_Feedrate' not in df.columns:
            continue

        X_raw = df.drop(columns=['Time(s)', 'Actual_Feedrate'], errors='ignore')
        actual_feedrate = df['Actual_Feedrate'].values

        # Scale input
        X_scaled = x_scaler.transform(X_raw).astype(np.float32)
        num_features = X_scaled.shape[1]

        # Padding
        past_padding = np.zeros((LOOK_BACK, num_features), dtype=np.float32)
        future_padding = np.zeros((LOOK_AHEAD, num_features), dtype=np.float32)
        X_padded = np.vstack((past_padding, X_scaled, future_padding))

        # Sequences
        total_rows = len(X_scaled)
        X_sequences = np.zeros((total_rows, LOOK_BACK + 1 + LOOK_AHEAD, num_features), dtype=np.float32)

        for i in range(total_rows):
            center = i + LOOK_BACK
            X_sequences[i] = X_padded[center - LOOK_BACK : center + LOOK_AHEAD + 1]

        # Predict
        predicted_scaled = model.predict(X_sequences, batch_size=2048, verbose=0)
        predicted_feedrate = y_scaler.inverse_transform(predicted_scaled).flatten()
        predicted_feedrate = np.maximum(predicted_feedrate, 0.001)  # Cegah devide by zero

        # --- KALKULASI WAKTU MACHINING ---
        # Actual Time = Total rows * 4ms
        actual_time = total_rows * CYCLE_TIME

        # Predicted Time:
        # Jarak per blok = Actual_Feedrate * 4ms
        # Prediksi Waktu per blok = Jarak / Predicted_Feedrate
        distance_per_block = actual_feedrate * CYCLE_TIME
        predicted_time_per_block = distance_per_block / predicted_feedrate
        predicted_time = np.sum(predicted_time_per_block)

        # Accuracy
        error = abs(actual_time - predicted_time)
        accuracy = max(0.0, 100 - ((error / actual_time) * 100))

        # Simpan total
        total_actual_time_all += actual_time
        total_predicted_time_all += predicted_time

        # Print row
        # Memotong nama file jika terlalu panjang agar tabel rapi
        display_name = file_name if len(file_name) <= 33 else file_name[:30] + "..."
        print(f"{display_name:<35} | {actual_time:<15.2f} | {predicted_time:<18.2f} | {accuracy:<15.2f}")

    print("-" * 90)

    # Calculate Overall Accuracy
    total_error = abs(total_actual_time_all - total_predicted_time_all)
    overall_accuracy = max(0.0, 100 - ((total_error / total_actual_time_all) * 100))

    print(f"{'OVERALL TOTAL':<35} | {total_actual_time_all:<15.2f} | {total_predicted_time_all:<18.2f} | {overall_accuracy:<15.2f}")
    print("[+] Evaluasi Selesai.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Penggunaan: python evaluate_batch_time.py <direktori_dataset_test>")
    else:
        test_directory = sys.argv[1]
        evaluate_directory(test_directory)
