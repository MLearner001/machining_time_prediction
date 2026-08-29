import pandas as pd
import numpy as np
import os
import time


def convert_angles_to_vector_dmg(angle_b_array, angle_c_array):
    b, c = np.radians(angle_b_array), np.radians(angle_c_array)
    a3 = np.sin(b) * np.cos(c)
    b3 = -np.sin(b) * np.sin(c)
    c3 = np.cos(b)
    return np.clip(a3, -1.0, 1.0), np.clip(b3, -1.0, 1.0), np.clip(c3, -1.0, 1.0)


def synchronize_cnc_20_slots_turbo(trace_path, gcode_path, output_path):
    with open(trace_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    header_idx = next(idx for idx, line in enumerate(lines) if line.lower().startswith('time,'))
    df_trace = pd.read_csv(trace_path, skiprows=header_idx)
    df_gcode = pd.read_csv(gcode_path)

    # RAM Dict Caching (Kebal Data Loss karena ID di Skrip 1 sudah dilinearkan unik)
    gcode_dict = df_gcode.set_index('Block_ID').to_dict(orient='index')

    raw_trace_blocks = df_trace['f1\\s1'].values.astype(int)
    raw_trace_blocks[raw_trace_blocks < 0] = 0

    # PENYEMBUHAN ROLLOVER KRONOLOGIS TRACE
    chronological_trace_blocks = np.zeros_like(raw_trace_blocks)
    rollover_accumulator = 0
    prev_raw_val = -1
    for k in range(len(raw_trace_blocks)):
        current_raw_val = raw_trace_blocks[k]
        if prev_raw_val > 0 and current_raw_val < prev_raw_val and (prev_raw_val - current_raw_val) > 50000:
            rollover_accumulator += 100000
        prev_raw_val = current_raw_val
        chronological_trace_blocks[k] = current_raw_val + rollover_accumulator

    act_x, act_y, act_z = df_trace['f2\\s2'].values, df_trace['f3\\s3'].values, df_trace['f4\\s4'].values
    act_b, act_c, act_feed = df_trace['f5\\s5'].values, df_trace['f6\\s6'].values, df_trace['f7\\s7'].values
    trace_times = df_trace['time'].values

    act_vec_a, act_vec_b, act_vec_c = convert_angles_to_vector_dmg(act_b, act_c)

    total_rows = len(df_trace)
    MAX_SLOTS, NUM_FEATURES_PER_SLOT = 20, 18
    slot_features_matrix = np.zeros((total_rows, MAX_SLOTS * NUM_FEATURES_PER_SLOT))
    num_blocks_column = np.ones(total_rows, dtype=int)

    for i in range(total_rows):
        current_block = chronological_trace_blocks[i]
        blocks_in_window = [current_block]
        num_blocks_column[i] = len(blocks_in_window)

        row_idx = 0
        for b_id in blocks_in_window[:MAX_SLOTS]:
            if b_id in gcode_dict:
                g = gcode_dict[b_id]
                start_slot = row_idx * NUM_FEATURES_PER_SLOT
                slot_features_matrix[i, start_slot:start_slot + NUM_FEATURES_PER_SLOT] = [
                    g['Cmd_F'], g['Cmd_S'], g['Is_G01'], g['Is_G02'], g['Is_G03'], g['Is_Traori'],
                    g['C832_Tol'], g['C832_Mode'], g['C800_RotX'], g['C800_RotY'], g['C800_RotZ'],
                    g['Is_Macro'],  # <-- Sukses dijajarkan secara horizontal ke samping!
                    g['Tgt_X'], g['Tgt_Y'], g['Tgt_Z'], g['Tgt_A'], g['Tgt_B'], g['Tgt_C']
                ]
                row_idx += 1

    columns = ['Time(s)', 'Num_Blocks']
    for s in range(1, MAX_SLOTS + 1):
        columns.extend([f'B{s}_Cmd_F', f'B{s}_Cmd_S', f'B{s}_Is_G01', f'B{s}_Is_G02', f'B{s}_Is_G03', f'B{s}_Is_Traori',
                        f'B{s}_C832_Tol', f'B{s}_C832_Mode', f'B{s}_C800_RotX', f'B{s}_C800_RotY', f'B{s}_C800_RotZ',
                        f'B{s}_Is_Macro', f'B{s}_Tgt_X', f'B{s}_Tgt_Y', f'B{s}_Tgt_Z', f'B{s}_Tgt_A', f'B{s}_Tgt_B',
                        f'B{s}_Tgt_C'])
    columns.extend(['Actual_X', 'Actual_Y', 'Actual_Z', 'Actual_A', 'Actual_B', 'Actual_C', 'Actual_Feedrate'])

    df_features = pd.DataFrame(slot_features_matrix)
    df_final = pd.concat([pd.DataFrame({'Time(s)': trace_times, 'Num_Blocks': num_blocks_column}), df_features], axis=1)
    df_final['Actual_X'], df_final['Actual_Y'], df_final['Actual_Z'] = act_x, act_y, act_z
    df_final['Actual_A'], df_final['Actual_B'], df_final['Actual_C'] = act_vec_a, act_vec_b, act_vec_c
    df_final['Actual_Feedrate'] = act_feed

    df_final.columns = columns
    df_final.to_csv(output_path, index=False)


def process_batch_synchronization():
    trace_dir, parsed_gcode_dir, output_dir = "./trace_sinutrain_data", "./gcode_parsed_batch", "./final_dataset_batch"
    if not os.path.exists(trace_dir): os.makedirs(trace_dir); return
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    trace_files = [f for f in os.listdir(trace_dir) if f.lower().endswith('.csv')]

    for idx, file_name in enumerate(trace_files, start=1):
        base_name = os.path.splitext(file_name)[0]
        gcode_partner_name = f"{base_name}_parsed.csv"
        gcode_partner_path = os.path.join(parsed_gcode_dir, gcode_partner_name)

        if os.path.exists(gcode_partner_path):
            start_time = time.time()
            print(f"    [{idx}/{len(trace_files)}] Menyandingkan Pasangan Turbo: {file_name}")
            try:
                synchronize_cnc_20_slots_turbo(os.path.join(trace_dir, file_name), gcode_partner_path,
                                               os.path.join(output_dir, f"{base_name}_sync_final.csv"))
                print(f"    [✔] Sukses! Waktu Proses: {time.time() - start_time:.2f} detik.")
            except Exception as e:
                print(f"    [❌] Gagal menyatukan {file_name}: {e}")
    print("\n[✔] Proses sinkronisasi batch selesai tanpa lemot!")


if __name__ == "__main__":
    process_batch_synchronization()
