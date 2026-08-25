import pandas as pd
df = pd.read_csv('real_merchants.csv')
print(df[df['merchant_name'].str.contains('Walmart', case=False, na=False)])