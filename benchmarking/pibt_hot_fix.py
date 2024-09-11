import os
import pandas as pd

# Define the input and output directories
input_dir = 'benchmarking/old_benchmarking/8_agents_results/'
output_dir = 'pibt_out/'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each CSV file in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith('.csv'):
        # Full path to the input file
        input_file_path = os.path.join(input_dir, file_name)
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file_path)
        
        # Filter rows where Program is GNNMAPF
        filtered_df = df[df['Program'] == 'GNNMAPF']
        
        # Change Program column to 'PIBT'
        filtered_df['Program'] = 'PIBT'
        
        # Generate new file name with '_pibt' appended
        new_file_name = file_name.replace('.csv', '_pibt.csv')
        output_file_path = os.path.join(output_dir, new_file_name)
        
        # Save the updated DataFrame to the new CSV file
        filtered_df.to_csv(output_file_path, index=False)

print("Processing complete. Files have been saved to the 'pibt_out' directory.")