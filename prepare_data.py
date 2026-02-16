import pandas as pd

#loading dataset
df=pd.read_csv('real_merchants.csv')

print(f"No of dataset: {len(df)}")

#check for columns
print(f"Columns: {df.columns.tolist()}")

# we dont need mcc_code
if 'mcc_code' in df.columns:
    df=df.drop('mcc_code', axis=1)

#check for missing values
missing = (df.isnull().sum())
if missing.any():
    df=df.dropna()

#show category breakdown
print(df['category'].value_counts())

#save
df.to_csv('merchants_categories.csv', index=False)