import pandas as pd
import os

# Load the CSV file into a DataFrame
df = pd.read_csv('merged_sheet.csv')

# Read the list of filenames from ocrFileNames.txt
with open('ocrImgFileNames.txt', 'r') as file:
    filenames = [os.path.basename(line.strip()) for line in file]

# Create a dictionary to store filename-plate pairs
results = {}

# Iterate through the DataFrame and populate the results dictionary
for index, row in df.iterrows():
    filename = os.path.basename(row['file'])  # Extract only the filename
    for name in filenames:
        if name in filename:
            results[filename] = row['plate']
            break  # Stop searching if a match is found

# Write the results to original_results.txt in the correct order
with open('original_results.txt', 'w') as file:
    for filename in filenames:
        plate = results.get(filename, 'Plate not found')
        file.write(f'{plate}\n')

print("Plate numbers have been written to original_results.txt in the correct order")
