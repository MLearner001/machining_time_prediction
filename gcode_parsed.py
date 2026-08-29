import re
import pandas as pd
import os


def parse_siemens_gcode_ultimate(nc_file_path, output_csv_path):
    """Mengurai file G-code Siemens secara batch dengan sensor MCALL, CYCLE832, dan CYCLE800."""
    r_parameters = {}
    r_assign_regex = re.compile(r'([Rr]\d+)\s*=\s*(-?\d*\.?\d+)')
    c832_regex = re.compile(r'CYCLE832\s*\(\s*(\d+\.?\d*)\s*,\s*_([A-Z]+)', re.IGNORECASE)
    c800_regex = re.compile(r'CYCLE800\s*\((.*?)\)', re.IGNORECASE)

    coord_pattern = re.compile(r'\b([XYZFGNST])\s*=?\s*(-?\d*\.?\d+)\b')
    vector_pattern = re.compile(r'\b([ABC]3)\s*=\s*(-?\d*\.?\d+)\b')

    with open(nc_file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            line = line.split(';')[0].strip()
            r_matches = r_assign_regex.findall(line)
            for r_var, r_val in r_matches:
                r_parameters[r_var.upper()] = float(r_val)

    current_g_mode, current_feed, current_spindle = "G01", 1000.0, 0.0
    last_x, last_y, last_z = 0.0, 0.0, 0.0
    last_v_a, last_v_b, last_v_c = 0.0, 0.0, 1.0
    is_traori_active, c832_tolerance, c832_mode_id = 0, 0.100, 0
    c800_rot_x, c800_rot_y, c800_rot_z = 0.0, 0.0, 0.0

    # KUNCI MEMORI MODAL STATUS MCALL
    is_macro_active = 0

    prev_raw_block_id = -1
    rollover_accumulator = 0
    parsed_blocks = []

    with open(nc_file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line_num, line in enumerate(file, start=1):
            line = line.split(';')[0].strip()
            if not line: continue
            if "G153" in line.upper() or "SUPA" in line.upper(): continue

            # SAKELAR MCALL MACRO DETECTOR
            if "MCALL" in line.upper():
                cleaned_mcall = re.sub(r'\s+', ' ', line.upper()).strip()
                tokens = cleaned_mcall.split()
                # Jika baris hanya tertulis MCALL murni atau Nxxx MCALL murni
                if len(tokens) == 1 or (len(tokens) == 2 and tokens[0].startswith('N') and tokens[1] == 'MCALL'):
                    is_macro_active = 0
                else:
                    is_macro_active = 1

                # Masukkan baris pendaftaran sebagai jangkar lini waktu
                try:
                    n_match = re.search(r'\bN(\d+)\b', line, re.IGNORECASE)
                    b_id = int(n_match.group(1)) + rollover_accumulator if n_match else line_num + rollover_accumulator
                except Exception:
                    b_id = line_num + rollover_accumulator

                parsed_blocks.append([
                    b_id, current_feed, current_spindle,
                    1 if current_g_mode == "G01" else 0, 0, 0,
                    is_traori_active, c832_tolerance, c832_mode_id, c800_rot_x, c800_rot_y, c800_rot_z,
                    is_macro_active, last_x, last_y, last_z, last_v_a, last_v_b, last_v_c
                ])
                continue

            if "CYCLE800" in line.upper():
                c800_match = c800_regex.search(line)
                if c800_match:
                    args_str = c800_match.group(1)
                    if args_str.strip():
                        args = [a.replace('"', '').strip() for a in args_str.split(',')]
                        c800_rot_x, c800_rot_y, c800_rot_z = 0.0, 0.0, 0.0
                        try:
                            if len(args) >= 11:
                                c800_rot_x, c800_rot_y, c800_rot_z = float(args[7]), float(args[8]), float(args[9])
                            elif len(args) >= 7:
                                c800_rot_x, c800_rot_y, c800_rot_z = float(args[4]), float(args[5]), float(args[6])
                        except (ValueError, IndexError):
                            pass
                    else:
                        c800_rot_x, c800_rot_y, c800_rot_z = 0.0, 0.0, 0.0
                continue

            if "CYCLE832" in line.upper():
                c832_match = c832_regex.search(line)
                if c832_match:
                    c832_tolerance = float(c832_match.group(1))
                    m_str = c832_match.group(2).upper()
                    c832_mode_id = 1 if m_str == "ROUGHING" else (2 if m_str == "SEMIFINISH" else 3)
                continue

            if "TRAORI" in line.upper(): is_traori_active = 1; continue
            if "TRAFOOF" in line.upper(): is_traori_active = 0; continue
            if '=' in line and line.startswith('N') and any(
                r in line.upper() for r in ['R102', 'R103', 'R104']) and 'G' not in line.upper(): continue

            matches_coord = coord_pattern.findall(line)
            matches_vector = vector_pattern.findall(line)
            if not matches_coord and not matches_vector: continue

            block_dict = {k.upper(): v for k, v in matches_coord}
            vector_dict = {k.upper(): v for k, v in matches_vector}

            try:
                raw_block_id = int(float(block_dict.get('N', line_num)))
                if prev_raw_block_id > 0 and raw_block_id < prev_raw_block_id and (
                        prev_raw_block_id - raw_block_id) > 50000:
                    rollover_accumulator += 100000
                prev_raw_block_id = raw_block_id
                chronological_block_id = raw_block_id + rollover_accumulator
            except ValueError:
                chronological_block_id = line_num + rollover_accumulator

            if 'G' in block_dict:
                g_val = block_dict['G']
                if g_val in ['0', '1', '2', '3']: current_g_mode = f"G0{int(float(g_val))}"
            if 'F' in block_dict:
                f_val = str(block_dict['F']).upper()
                current_feed = r_parameters[f_val] if f_val in r_parameters else float(f_val)
            if 'S' in block_dict:
                try:
                    current_spindle = float(block_dict['S'])
                except ValueError:
                    pass

            tgt_x = float(block_dict.get('X', last_x))
            tgt_y = float(block_dict.get('Y', last_y))
            tgt_z = float(block_dict.get('Z', last_z))
            last_x, last_y, last_z = tgt_x, tgt_y, tgt_z

            tgt_a = float(vector_dict.get('A3', last_v_a))
            tgt_b = float(vector_dict.get('B3', last_v_b))
            tgt_c = float(vector_dict.get('C3', last_v_c))
            last_v_a, last_v_b, last_v_c = tgt_a, tgt_b, tgt_c

            parsed_blocks.append([
                chronological_block_id, current_feed, current_spindle,
                1 if current_g_mode == "G01" else 0, 1 if current_g_mode == "G02" else 0,
                1 if current_g_mode == "G03" else 0,
                is_traori_active, c832_tolerance, c832_mode_id, c800_rot_x, c800_rot_y, c800_rot_z,
                is_macro_active, tgt_x, tgt_y, tgt_z, tgt_a, tgt_b, tgt_c
            ])

    columns = ['Block_ID', 'Cmd_F', 'Cmd_S', 'Is_G01', 'Is_G02', 'Is_G03', 'Is_Traori', 'C832_Tol', 'C832_Mode',
               'C800_RotX', 'C800_RotY', 'C800_RotZ', 'Is_Macro', 'Tgt_X', 'Tgt_Y', 'Tgt_Z', 'Tgt_A', 'Tgt_B', 'Tgt_C']
    pd.DataFrame(parsed_blocks, columns=columns).to_csv(output_csv_path, index=False)


def main():
    input_folder, output_folder = "./input_nc_data", "./gcode_parsed_batch"
    if not os.path.exists(input_folder): os.makedirs(input_folder); return
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    nc_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.nc', '.txt', '.tap', '.mpf'))]
    print(f"[+] Menemukan {len(nc_files)} file G-code valid.")
    for idx, file_name in enumerate(nc_files, start=1):
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_parsed.csv")
        print(f"    [{idx}/{len(nc_files)}] Parsing: {file_name}")
        parse_siemens_gcode_ultimate(os.path.join(input_folder, file_name), output_file_path)
    print("\n[✔] Batch parsing selesai dilaksanakan!")


if __name__ == "__main__":
    main()
