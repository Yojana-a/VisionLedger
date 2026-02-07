import pandas as pd

print("Loading dataset...")

#load the csv file
df =  pd.read_csv('fraudTrain.csv')

print(f"Loaded {len(df)} transactions")# loads xxxx no of transactions

#extracting merchant and categories column
training_data = df[['merchant', 'category']].copy() #.copy( makes a seperate copy so we dont change the original

print(f"Original Categories: {training_data['category'].unique().tolist()}")
#.unique() identifies distinct values and .tolist() converts it to a list 

#remove suffixes like _pos
training_data['category'] = training_data['category'].str.replace('_pos', '', regex=False)# when regex=false turns off pattern language and goes for literal search
training_data['category'] = training_data['category'].str.replace('_net', '', regex=False)

#map cleaned categories to standard names
category_mapping = {
    'grocery': 'Groceries',
    'gas_transport': 'Transportation',
    'shopping': 'Shopping',
    'food_dining': 'Food & Dining',
    'entertainment': 'Entertainment',
    'health_fitness': 'Healthcare',
    'home': 'Shopping',
    'kids_pets': 'Shopping',
    'personal_care': 'Healthcare',
    'misc': 'Other',
    'travel': 'Transportation'
}

#apply the mapping
training_data['category']=training_data['category'].map(category_mapping)
# while modifying a specific column

#removing duplicates
training_data =training_data.drop_duplicates(subset=['merchant'], keep='first')
#while removing an entire row (using only training_data)

#clean merchants name
training_data['merchant'] = training_data['merchant'].str.replace('fraud_', '', regex=False)

training_data.columns=['merchant_name', 'category']

print(f"Total unique merchnts: {len(training_data)}")
print(training_data['category'].value_counts())

#save to csv
training_data.to_csv("merchant_categories.csv", index=False)
print("\n Saved to merchant_categories.csv")
