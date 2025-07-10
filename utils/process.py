import os
import pandas as pd
import ast
import re

def parse_any_list(s):
    """General parsing function, handles the following formats:
    1. Standard list [val1, val2, val3]
    2. array format [array([val1]), array([val2]), array([val3])]
    3. np.float64 format [np.float64(val1), np.float64(val2), np.float64(val3)]
    """
    try:
        # First, try to parse as a standard list directly
        try:
            return ast.literal_eval(s.strip())
        except:
            pass

        # Handle array format
        array_matches = re.findall(r'array\(\[([\d.]+)\]\)', s)
        if array_matches:
            return [float(x) for x in array_matches]

        # Handle np.float64 format
        float64_matches = re.findall(r'np\.float64\(([\d.]+)\)', s)
        if float64_matches:
            return [float(x) for x in float64_matches]

        # Handle mixed format
        mixed_matches = re.findall(r'(?:array\(\[([\d.]+)\]\)|np\.float64\(([\d.]+)\)|([\d.]+))', s)
        if mixed_matches:
            values = []
            for match in mixed_matches:
                # match is a tuple, take the first non-empty value
                val = next(x for x in match if x)
                values.append(float(val))
            return values

        print(f"Unparseable format: {s}")
        return []
    except Exception as e:
        print(f"Parsing error: {s} (Error: {str(e)})")
        return []

# Get all subfolders
folders = [f for f in os.listdir() if os.path.isdir(f) and not f.startswith('.')]

results = []

for folder in folders:
    file_path = os.path.join(folder, "gca_GT_NPDC_market.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)

            # Parse all columns
            test_mse = parse_any_list(df['test_mse'].iloc[0])
            test_mae = parse_any_list(df['test_mae'].iloc[0])
            test_rmse = parse_any_list(df['test_rmse'].iloc[0])
            test_mape = parse_any_list(df['test_mape'].iloc[0])
            test_mse_per_target = parse_any_list(df['test_mse_per_target'].iloc[0])

            # Validate data length
            lengths = {
                'test_mse': len(test_mse),
                'test_mae': len(test_mae),
                'test_rmse': len(test_rmse),
                'test_mape': len(test_mape),
                'test_mse_per_target': len(test_mse_per_target)
            }

            if not all(l == 3 for l in lengths.values()):
                print(f"Warning: Data length mismatch in {folder}: {lengths}")
                continue

            # Create a row for each model
            models = ['GRU','LSTM', 'Transformer']
            for i, model in enumerate(models):
                try:
                    results.append({
                        'folder': folder,
                        'model': model,
                        'test_mse': test_mse[i],
                        'test_mae': test_mae[i],
                        'test_rmse': test_rmse[i],
                        'test_mape': test_mape[i],
                        'test_mse_per_target': test_mse_per_target[i]
                    })
                except IndexError:
                    print(f"Warning: Missing data for model {model} in {folder}")
                    continue

        except Exception as e:
            print(f"Error processing {folder}: {str(e)}")
            continue

# Save results
if results:
    final_df = pd.DataFrame(results)
    # Sort by folder and model
    final_df = final_df.sort_values(by=['folder', 'model'])

    # Export using UTF-8 encoding, ensure correct handling of Chinese and special characters
    final_df.to_csv('merged_results.csv',
                   index=False,
                   encoding='utf-8-sig',  # Use UTF-8 with BOM so Excel recognizes it correctly
                   float_format='%.8f')  # Control floating-point precision

    print(f"\nMerge complete, processed {len(folders)} folders in total")
    print(f"Successfully parsed {len(results)} records (expected {len(folders)*3} records)")
    print("UTF-8 encoded results saved to merged_results.csv")

    # Print folders that failed to process
    processed_folders = set(df['folder'] for df in results)
    failed_folders = set(folders) - processed_folders
    if failed_folders:
        print("\nThe following folders failed to process:")
        for f in sorted(failed_folders):
            print(f" - {f}")
else:
    print("No valid data found")