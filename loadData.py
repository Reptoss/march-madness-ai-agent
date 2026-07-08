import os
import kagglehub
import pandas as pd

def fetch_march_madness_data():
    print("Connecting to Kaggle...")
    data_dir = kagglehub.dataset_download("nishaanamin/march-madness-data")
    
    print(func_name := f"Success! Files downloaded to localized cache: {data_dir}")
    print("Files available:")
    
    files = os.listdir(data_dir)
    for file in files:
        print(f" - {file}")
        
    target_file = "teams.csv" 
    
    if target_file in files:
        file_path = os.path.join(data_dir, target_file)
        df = pd.read_csv(file_path)
        print(f"\nSuccessfully loaded {target_file} into memory!")
        print(df.head()) # Preview the first 5 rows
        return df
    else:
        print(f"\nCould not find {target_file} directly, but data is ready in cache.")
        return None

if __name__ == "__main__":
    df = fetch_march_madness_data()
