import pandas as pd
import os
import glob

def merge_csv_files():
    """
    Merges the second column of all CSV files in the current directory into a single CSV.
    Each column in the output is named after the source file (without extension).
    """
    # Get all csv files in the current directory
    csv_files = sorted(glob.glob("*.csv"))
    
    # Exclude the output file if it already exists
    output_filename = "merged_reward.csv"
    csv_files = [f for f in csv_files if f != output_filename]
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    merged_df = None
    
    for file_path in csv_files:
        # Get filename without extension for the column name
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if there's a Step column and a data column
            if df.shape[1] >= 2:
                # Keep only Step (column 0) and Data (column 1)
                temp_df = df.iloc[:, [0, 1]].copy()
                temp_df.columns = ['Step', file_name]
                
                if merged_df is None:
                    merged_df = temp_df
                else:
                    # Merge on 'Step' to align values
                    merged_df = pd.merge(merged_df, temp_df, on='Step', how='outer')
                
                print(f"Processed {file_path}")
            else:
                print(f"Warning: {file_path} does not have enough columns. Skipping.")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if merged_df is not None:
        # Sort by Step to ensure chronological order
        merged_df = merged_df.sort_values(by='Step').reset_index(drop=True)
        
        # Save to csv
        merged_df.to_csv(output_filename, index=False)
        print(f"\nSuccessfully merged files into {output_filename}")
    else:
        print("\nNo data was collected. Check if the CSV files are valid.")

if __name__ == "__main__":
    merge_csv_files()

