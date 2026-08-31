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

def evaluate_directory(nc_dir, parsed_dir, trace_dir, model_path="bi_lstm_lookahead_model.keras"):
    # Ensure all directories exist
    for directory in [nc_dir, parsed_dir, trace_dir]:
        if not os.path.exists(directory):
            print(f"[!] Direktori {directory} tidak ditemukan.")
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

    # List NC files as the anchor
    nc_files = [f for f in os.listdir(nc_dir) if f.endswith('.mpf') or f.endswith('.nc')]
    if not nc_files:
        print(f"[!] Tidak ada file NC di {nc_dir}")
        return

    print(f"\n[+] Ditemukan {len(nc_files)} file NC. Memulai cross-evaluasi time blocks...\n")
    print(f"{'NC File Name':<35} | {'Actual Time (s)':<15} | {'Predicted Time (s)':<18} | {'Accuracy (%)':<15}")
    print("-" * 90)

    total_actual_time_all = 0.0
    total_predicted_time_all = 0.0

    for nc_name in sorted(nc_files):
        nc_base = os.path.splitext(nc_name)[0]

        # We need the synced final dataset feature file for Neural Net predictions
        # Note: The synchronization pipeline outputs to `final_dataset_batch` by default.
        # Since this script needs to load the scaled dataset features to predict, we will assume
        # the synced final CSV has the same base name.
        synced_csv_path = os.path.join("./final_dataset_batch", f"{nc_base}.csv")

        # And we need the SinuTrain trace to get True Actual Machining Time
        # Looking up the trace matching the NC file
        trace_candidates = [f for f in os.listdir(trace_dir) if nc_base in f and f.endswith('.csv')]
        if not trace_candidates or not os.path.exists(synced_csv_path):
            # Skip if we don't have the fully synced dataset for predictions
            continue

        trace_path = os.path.join(trace_dir, trace_candidates[0])

        # 1. ACTUAL TIME from SinuTrain trace (Number of rows * 4ms)
        # Bypassing the header 'Time,' row dynamically as done in pipeline
        with open(trace_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        try:
            header_idx = next(idx for idx, line in enumerate(lines) if line.lower().startswith('time,'))
            df_trace = pd.read_csv(trace_path, skiprows=header_idx)
            actual_time = len(df_trace) * CYCLE_TIME
        except Exception:
            continue

        # Load the Parsed G-code to get TRUE NC block physical distances
        # Sesuai folder parsed_dir dan format penamaan `_parsed.csv`
        parsed_path = os.path.join(parsed_dir, f"{nc_base}_parsed.csv")
        if not os.path.exists(parsed_path):
            # Coba tanpa _parsed
            parsed_path = os.path.join(parsed_dir, f"{nc_base}.csv")
            if not os.path.exists(parsed_path):
                continue

        df_parsed = pd.read_csv(parsed_path)

        # Kalkulasi Jarak Pergerakan Spindle per Blok G-code (Euclidean 3D Distance)
        try:
            x_nc = df_parsed['Tgt_X'].values
            y_nc = df_parsed['Tgt_Y'].values
            z_nc = df_parsed['Tgt_Z'].values
            block_ids = df_parsed['Block_ID'].values

            # Hitung delta pergeseran NC antar blok berurutan
            dx = np.diff(x_nc, prepend=x_nc[0])
            dy = np.diff(y_nc, prepend=y_nc[0])
            dz = np.diff(z_nc, prepend=z_nc[0])

            # Phytagoras (Jarak sesungguhnya per blok dalam mm)
            nc_block_distances = np.sqrt(dx**2 + dy**2 + dz**2)

        except KeyError:
            print(f"[!] Kolom Tgt_X/Y/Z tidak ditemukan di {parsed_path}. Melewati...")
            continue

        # 2. PREDICTED TIME from Neural Net outputs
        # Load the fully synced features for prediction
        df_features = pd.read_csv(synced_csv_path)

        # Caching the Num_Blocks for prediction aggregation
        if 'Num_Blocks' not in df_features.columns:
            print(f"[!] Kolom Num_Blocks tidak ditemukan di {synced_csv_path}. Melewati...")
            continue
        synced_block_ids = df_features['Num_Blocks'].values

        X_raw = df_features.drop(columns=['Time(s)', 'Actual_Feedrate'], errors='ignore')

        # Scale input
        X_scaled = x_scaler.transform(X_raw).astype(np.float32)
        num_features = X_scaled.shape[1]

        # Padding (mencegah OOM np.zeros besar dengan loop prediksi)
        past_padding = np.zeros((LOOK_BACK, num_features), dtype=np.float32)
        future_padding = np.zeros((LOOK_AHEAD, num_features), dtype=np.float32)
        X_padded = np.vstack((past_padding, X_scaled, future_padding))

        # Predict using Generator to strictly prevent RAM OOM ballooning
        def data_generator():
            for i in range(len(X_scaled)):
                center = i + LOOK_BACK
                yield X_padded[center - LOOK_BACK : center + LOOK_AHEAD + 1]

        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=tf.TensorSpec(shape=(LOOK_BACK + 1 + LOOK_AHEAD, num_features), dtype=tf.float32)
        ).batch(2048).prefetch(tf.data.AUTOTUNE)

        predicted_scaled = model.predict(dataset, verbose=0)
        predicted_feedrate = y_scaler.inverse_transform(predicted_scaled).flatten()
        predicted_feedrate = np.maximum(predicted_feedrate, 1e-6)

        # --- MENGGABUNGKAN JARAK NC (G-CODE) DENGAN PREDIKSI NN (SINKRON) ---
        # Karena NN memprediksi feedrate setiap 4ms, dan jarak (distance) dihitung per blok G-code,
        # kita harus merata-ratakan predicted_feedrate (atau mengambil mean) untuk setiap Blok G-code!

        # Kumpulkan Prediksi NN ke dalam dictionary berdasarkan Block ID
        # Kita menggunakan pandas groupby agar sangat cepat
        df_preds = pd.DataFrame({'Block_ID': synced_block_ids, 'Predicted_F': predicted_feedrate})

        # Agregasi mean (rata-rata kecepatan aktual yang dieksekusi pada blok tersebut)
        mean_feedrate_per_block = df_preds.groupby('Block_ID')['Predicted_F'].mean().to_dict()

        # Hitung waktu per blok NC
        predicted_time_per_block = []
        for i, b_id in enumerate(block_ids):
            dist = nc_block_distances[i]
            if dist == 0:
                # Tidak ada perpindahan (G04 Dwell atau murni rotasi)
                continue

            # Dapatkan feedrate yang diprediksi untuk blok ini, jika tidak ada, fallback ke speed sangat kecil
            pred_f = mean_feedrate_per_block.get(b_id, 1e-6)

            # Feedrate mm/min -> mm/s
            pred_f_sec = pred_f / 60.0

            block_time = dist / pred_f_sec
            predicted_time_per_block.append(block_time)

        predicted_time = np.sum(predicted_time_per_block)

        # Accuracy Calculation
        error = abs(actual_time - predicted_time)
        accuracy = max(0.0, 100 - ((error / actual_time) * 100))

        # Simpan total
        total_actual_time_all += actual_time
        total_predicted_time_all += predicted_time

        # Print row
        display_name = nc_name if len(nc_name) <= 33 else nc_name[:30] + "..."
        print(f"{display_name:<35} | {actual_time:<15.2f} | {predicted_time:<18.2f} | {accuracy:<15.2f}")

    print("-" * 90)

    # Calculate Overall Accuracy
    total_error = abs(total_actual_time_all - total_predicted_time_all)
    overall_accuracy = max(0.0, 100 - ((total_error / total_actual_time_all) * 100))

    print(f"{'OVERALL TOTAL':<35} | {total_actual_time_all:<15.2f} | {total_predicted_time_all:<18.2f} | {overall_accuracy:<15.2f}")
    print("[+] Evaluasi Selesai.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Penggunaan: python evaluate_batch_time.py <nc_dir> <parsed_dir> <trace_dir>")
    else:
        nc_dir = sys.argv[1]
        parsed_dir = sys.argv[2]
        trace_dir = sys.argv[3]
        evaluate_directory(nc_dir, parsed_dir, trace_dir)
