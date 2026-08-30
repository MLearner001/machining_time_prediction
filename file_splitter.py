import os
import pandas as pd
import gc

def split_csv_files(input_dir="./final_dataset_batch", output_dir="./split_dataset_small", chunk_size=50000):
    """
    Reads large CSV files from input_dir and splits them into smaller chunks
    of exactly chunk_size rows each. Saves the micro-files to output_dir.
    Designed to bypass memory limits by chunking on the host I/O.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist. Please ensure it exists and contains the dataset.")
        return

    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in {input_dir}.")
        return

    print(f"Found {len(csv_files)} CSV files in {input_dir}. Starting splitting process...")

    for file_name in csv_files:
        file_path = os.path.join(input_dir, file_name)
        base_name = os.path.splitext(file_name)[0]
        print(f"Processing {file_name}...")

        try:
            # Using chunksize to avoid loading the entire file into RAM at once
            chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size)
            for i, chunk in enumerate(chunk_iterator):
                output_file_name = f"{base_name}_part_{i+1:04d}.csv"
                output_file_path = os.path.join(output_dir, output_file_name)
                chunk.to_csv(output_file_path, index=False)
                print(f"  Saved {output_file_name} ({len(chunk)} rows)")

                # Force garbage collection to prevent memory ballooning during split
                del chunk
                gc.collect()
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print("Splitting process completed successfully.")

if __name__ == "__main__":
    split_csv_files()
