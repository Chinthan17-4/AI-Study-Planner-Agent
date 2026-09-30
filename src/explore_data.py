import pandas as pd

df = pd.read_csv("data/study_data.csv")

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- DATASET INFO ---")
print(df.info())

print("\n--- STATISTICS ---")
print(df.describe())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())