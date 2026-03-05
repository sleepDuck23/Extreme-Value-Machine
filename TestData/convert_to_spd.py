import numpy as np
import argparse
import os

def convert_file_to_spd(input_path, output_path, window_size, epsilon=1e-5):
    """
    Reads a flat feature text file, converts each row to an SPD matrix using a 
    sliding window (Hankel matrix), flattens it, and saves it to a new text file.
    """
    print(f"Processing {input_path}...")
    
    # Load raw data (assumes format: label, feat1, feat2, ...)
    try:
        raw_data = np.loadtxt(input_path, delimiter=",", dtype=str)
    except FileNotFoundError:
        print(f"Error: Could not find the file {input_path}")
        return

    labels = raw_data[:, 0]
    features = raw_data[:, 1:].astype(float)
    
    num_samples = features.shape[0]
    num_features = features.shape[1]
    
    # Check if window size is valid
    if window_size >= num_features:
        print(f"Error: Window size ({window_size}) must be less than the number of features ({num_features}).")
        return

    num_rows = num_features - window_size + 1 
    
    with open(output_path, 'w') as f_out:
        for i in range(num_samples):
            vector = features[i]
            label = labels[i]
            
            # 1. Build the Hankel Matrix
            H = np.column_stack([vector[j : j + num_rows] for j in range(window_size)])
            
            # 2. Create the Covariance Matrix
            C = np.dot(H.T, H)
            
            # 3. Regularize to ensure it is strictly Positive Definite
            C_spd = C + epsilon * np.eye(window_size)
            
            # 4. Flatten the matrix to save it as a CSV row
            flat_spd = C_spd.flatten()
            
            # 5. Write to the new file: "label,val1,val2,..."
            line = f"{label}," + ",".join(map(str, flat_spd)) + "\n"
            f_out.write(line)
            
    print(f"Saved {num_samples} samples to {output_path}")
    print(f"New feature size per sample: {window_size * window_size} (from a {window_size}x{window_size} matrix)\n")

if __name__ == "__main__":
    
    WINDOW_SIZE = 4
    
    # Process the Training File
    convert_file_to_spd("train_synthetic.txt", f"train_synthetic_spd_w{WINDOW_SIZE}.txt", window_size=WINDOW_SIZE)
    
    # Process the Testing File
    convert_file_to_spd("test_synthetic.txt", f"test_synthetic_spd_w{WINDOW_SIZE}.txt", window_size=WINDOW_SIZE)