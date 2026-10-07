import pandas as pd
from sklearn.datasets import load_breast_cancer

def generate():
    """
    Generates the standard 569-row Wisconsin Diagnostic Breast Cancer dataset.
    This provides 30 real clinical features (radius, texture, smoothness, perimeter)
    and a binary diagnosis target (0: Benign, 1: Malignant).
    """
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    
    # Rename target column to a clean, intuitive name
    df.rename(columns={'target': 'diagnosis'}, inplace=True)
    
    output_filename = "real_cancer_dataset.csv"
    df.to_csv(output_filename, index=False)
    print(f"Dataset successfully created: {output_filename}")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    print("Target Column: 'diagnosis'")

if __name__ == "__main__":
    generate()