import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

print("Loading the training data")

#load the data
df=pd.read_csv("merchant_categories.csv")

print(f"Loading {len(df)} merchants")
print(f"Categories: {df['category'].unique().tolist()}")

#prepare data for machine learning
print("\n Preparing data for machine learning")

#X=input merchants name- what we use to predict
#y=Output(categories)-what we want to predict
X = df['merchant_name']
y=df['category']

#Convert text to numbers
#Ml cannot read texts only numbers

vectorizer = TfidfVectorizer (
    lowercase=True, # Converts to lowercase
    ngram_range=(1,2),#single words and word pairs(eg. "whole" "foods" and "whole foods")
    max_features=500 # keep top 500 most important patterns
)


