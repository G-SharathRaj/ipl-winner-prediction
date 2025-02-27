import pandas as pd

csv_path = "IPL_Matches_2008_2022.csv"

# Load the dataset
data = pd.read_csv(csv_path)

# Print column names
print("Columns in the dataset:\n", data.columns)
