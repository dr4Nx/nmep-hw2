import numpy as np
import pandas as pd
from pathlib import Path


def npytocsv(input_path):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    data = np.load(input_path)
    data = np.argmax(data, axis=1)
    df = pd.DataFrame(data, columns=["Category"])
    
    filename = "submission.csv"
    df.to_csv(filename, header=['Category'], index_label='Id')
    print(f"Saved submission file to {filename}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_path>")
    else:
        npytocsv(sys.argv[1])