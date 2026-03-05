import numpy as np
import argparse

def generate_gn_dataset(num_classes, train_samples, test_samples, seq_length=16):
    """
    Generates a synthetic dataset of Gaussian Noise time series.
    Each class has uniquely defined mean (mu) and standard deviation (sigma) parameters.
    """
    if num_classes > 26:
        print("Warning: Limiting to 26 classes to use A-Z labels.")
        num_classes = 26

    train_data = []
    test_data = []

    print(f"Generating data for {num_classes} classes...")
    
    for i in range(num_classes):
        # Assign a letter label (0 -> A, 1 -> B, etc.)
        label = chr(65 + i)
        
        # Define unique parameters for this class's Gaussian Noise
        # Example: Class A (mu=0, sigma=1.0), Class B (mu=1, sigma=1.5), etc.
        mu = float(i) 
        sigma = 1.0 + (i * 0.5) 
        
        print(f"  Class {label}: mu={mu:.1f}, sigma={sigma:.1f}")

        # Generate Training Samples
        for _ in range(train_samples):
            # Generate 16 random numbers from the Gaussian distribution
            series = np.random.normal(loc=mu, scale=sigma, size=seq_length)
            # Format as: "A,1.2,0.4,-0.1,..."
            row = f"{label}," + ",".join(map(str, series))
            train_data.append(row)
            
        # Generate Testing Samples
        for _ in range(test_samples):
            series = np.random.normal(loc=mu, scale=sigma, size=seq_length)
            row = f"{label}," + ",".join(map(str, series))
            test_data.append(row)

    # Save to text files
    train_file = "train_synthetic.txt"
    test_file = "test_synthetic.txt"
    
    with open(train_file, 'w') as f:
        f.write("\n".join(train_data) + "\n")
        
    with open(test_file, 'w') as f:
        f.write("\n".join(test_data) + "\n")

    print(f"\nSuccess! Created {train_file} ({num_classes * train_samples} samples)")
    print(f"Success! Created {test_file} ({num_classes * test_samples} samples)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Gaussian Noise time-series dataset.")
    parser.add_argument("--classes", type=int, default=5, help="Number of distinct classes to generate (max 26)")
    parser.add_argument("--train_samples", type=int, default=500, help="Number of training samples per class")
    parser.add_argument("--test_samples", type=int, default=100, help="Number of testing samples per class")
    
    args = parser.parse_args()
    
    generate_gn_dataset(args.classes, args.train_samples, args.test_samples)