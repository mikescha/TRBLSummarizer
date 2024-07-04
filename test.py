import pandas as pd
import numpy as np

# Example DataFrame
data = {'A': [np.nan, 10], 'B': [5, 11], 'C': [6, np.nan], 'D': [np.nan, np.nan], 'E': [7, 12]}
df = pd.DataFrame(data)

# Initialize lists to store results
first_columns = []
last_columns = []

# Iterate over each row
for idx, row in df.iterrows():
    first_column = None
    last_column = None
    nan_count = 0
    last_valid_col = None
    
    # Find the first column with value >= 4
    for col in row.index:
        if pd.notna(row[col]) and row[col] >= 4:
            first_column = col
            break
    
    # Find the last column with value >= 4 and no more than one preceding NaN
    for col in reversed(row.index):
        if col == first_column: 
            break

        if pd.notna(row[col]) and row[col] >= 4:
            if last_valid_col == None:
                last_valid_col = col
            nan_count = 0

        elif pd.isna(row[col]) or pd.notna(row[col]) and row[col] < 4:
            nan_count += 1
            if nan_count > 1:
                last_valid_col = None
    
    # If last_column is found but more than one NaN precedes it, reset last_column
#    if last_column and last_valid_column and last_valid_column != last_column:
#        last_column = None
    last_column = last_valid_col 

    # Append results to the lists
    first_columns.append(first_column)
    last_columns.append(last_column)

# Create a new DataFrame with the results
result_df = pd.DataFrame({
    'first_column': first_columns,
    'last_column': last_columns
})

print(result_df)
