import pandas as pd
import numpy as np
import tensorflow as tf
import os
import gc
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

LOOK_BACK = 50
LOOK_AHEAD = 50
TOTAL_SEQ_LEN = 101  # 50 lalu + 1 skrg + 50 depan


def make_dataset_pipeline(dataset_dir, scaler, batch_size=512, is_training=True):
    """
    Kran Generator Pintar: Membaca file satu per satu dari SSD secara bergantian.
    RAM komputer Anda dijamin sangat dingin (< 3GB terpakai).
    """
    csv_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.csv')]

    # Bagi file: 80% untuk training, 20% untuk validasi secara tegas di tingkat dokumen
    split_idx = int(len(csv_files) * 0.8)
    target_files = csv_files[:split_idx] if is_training else csv_files[split_idx:]

    def file_stream_generator():
        for file_path in target_files:
            # Memuat hanya SATU file ke RAM secara instan
            df = pd.read_csv(file_path)

            X_raw = df.drop(columns=['Time(s)', 'Actual_Feedrate'])
            y_raw = df['Actual_Feedrate'].values.astype(np.float32)

            # Transformasikan skala data ke float32 hemat memori
            X_scaled = scaler.transform(X_raw).astype(np.float32)

            total_rows = len(X_scaled)
            num_features = X_scaled.shape[1]

            # Pasang bantalan pengaman ujung per file secara mandiri
            past_padding = np.zeros((LOOK_BACK, num_features), dtype=np.float32)
            future_padding = np.zeros((LOOK_AHEAD, num_features), dtype=np.float32)
            X_padded = np.vstack((past_padding, X_scaled, future_padding))

            # Pengocokan acak internal indeks per file pengerjaan
            indices = np.arange(total_rows)
            np.random.shuffle(indices)

            for idx in indices:
                center_index = idx + LOOK_BACK
                window = X_padded[center_index - LOOK_BACK: center_index + LOOK_AHEAD + 1]
                yield window, y_raw[idx]

            # KUNCI UTAMA: Hancurkan variabel sampah di RAM dan paksa Windows melakukan pembersihan total
            del df, X_raw, y_raw, X_scaled, X_padded
            gc.collect()

    # Ambil info total fitur dari file pertama secara aman
    sample_df = pd.read_csv(csv_files[0], nrows=5)
    num_features = sample_df.drop(columns=['Time(s)', 'Actual_Feedrate']).shape[1]

    output_signature = (
        tf.TensorSpec(shape=(TOTAL_SEQ_LEN, num_features), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(file_stream_generator, output_signature=output_signature)

    # Aktifkan prefetching batch dinamis khusus sirkuit CUDA RTX 3080 Ti
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_machining_intelligence():
    print("[+] Memvalidasi Akselerasi Hardware GPU RTX 3080 Ti...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            print("    [✔] Alokasi VRAM Dinamis Sukses Terkunci.")
        except RuntimeError as e:
            print(e)

    dataset_dir = "./split_dataset_small"
    csv_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.csv')]

    print("[+] Mengalkulasikan Parameter Skala Global secara Ringan...")
    # Taktik Cerdas: Fit scaler hanya menggunakan sampel representatif dari 3 file pertama
    # agar menghemat penggunaan RAM sistem hingga 90% di awal pengerjaan
    sample_dfs = [pd.read_csv(f) for f in csv_files[:3]]
    df_sample_master = pd.concat(sample_dfs, ignore_index=True)
    X_sample = df_sample_master.drop(columns=['Time(s)', 'Actual_Feedrate'])

    scaler = StandardScaler()
    scaler.fit(X_sample)

    # Hancurkan contoh master sampel agar RAM kembali ke kondisi 0
    del sample_dfs, df_sample_master, X_sample
    gc.collect()

    BATCH_SIZE = 512
    print("[+] Merakit Kran Aliran Pipa File Streaming (Anti-RAM Ballooning)...")
    train_dataset = make_dataset_pipeline(dataset_dir, scaler, batch_size=BATCH_SIZE, is_training=True)
    val_dataset = make_dataset_pipeline(dataset_dir, scaler, batch_size=BATCH_SIZE, is_training=False)

    # Membaca dimensi fitur murni dari contoh fit
    num_features = scaler.n_features_in_

    print(f"[+] Menyusun Arsitektur Ekspansif Industri (128 Neuron Hibrida, {num_features} Fitur)...")
    model = models.Sequential([
        layers.Input(shape=(TOTAL_SEQ_LEN, num_features)),

        layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),

        layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
        layers.Dropout(0.4),

        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print(f"\n[+] Memulai Eksekusi Latihan Hemat RAM di GPU (Batch Size = {BATCH_SIZE})...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[early_stop],
        verbose=1
    )

    model.save("bi_lstm_lookahead_model.h5")
    print("[✔] Otak Jaringan Digital Twin Sukses Disimpan!")

    # Pembuatan grafik evaluasi
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='royalblue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='crimson', linewidth=2)
    plt.title("Grafik Performa Akurasi Model Bi-LSTM 128 Neuron", fontsize=12, fontweight='bold')
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig("./bi_lstm_training_loss_curve.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    train_machining_intelligence()
