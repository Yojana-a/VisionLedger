import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

#load the dataset
df=pd.read_csv('merchants_categories.csv')
print(f"Categories: {df['category'].unique().tolist()}")

#prepare input and output
X_text=df['merchant_name']
y=df['category']

#Vectorization
vectorizer=TfidfVectorizer(
    lowercase=True,
    ngram_range=(1,2),
    max_features=5000,
    sublinear_tf=True #smooths frequency differences
)

X=vectorizer.fit_transform(X_text)

print(f"Converted{len(X_text)} merchants into {X.shape[1]} numerical features")

#get unique merchants
unique_merchants=df['merchant_name'].unique()

#train_test_split
train_merchants, test_merchants = train_test_split(
    unique_merchants,
    test_size=0.2,
    random_state=42
)

#use mask to prevent data leakage as there may be a particular merchat like Starbucks in both train and test if we donot mask
train_mask=df['merchant_name'].isin(train_merchants).values
test_mask=df['merchant_name'].isin(test_merchants).values

X_train=X[train_mask]
X_test=X[test_mask]
y_train=y[train_mask]
y_test=y[test_mask]

print(f"Training merchants: {len(train_merchants)} unique merchants")
print(f"Test merchants: {len(test_merchants)} unique merchants")

#train model 
#training the ML model
print("Training the model")


model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    C=5.0
)

model.fit(X_train, y_train)

#Test the accuracy
print("\n Testing model's accuracy")

y_pred = model.predict(X_test) #making predictions on the test set(which the model has not seen)
accuracy= accuracy_score(y_test, y_pred)
print(f"{accuracy * 100:.1f}%")

#showing performance per category
print(classification_report(y_test,y_pred))

#testing with real examples
test_merchants=[
    "STARBUCKS #12345",
    "Whole Foods Market",
    "Shell Gas Station",
    "AMC Theater",
    "Amazon.com",
    "Verizon Wireless",
    "CVS Pharmacy",
    "Unknown Store"
]

for merchant in test_merchants:
    #Convert to numbers
    X_example = vectorizer.transform([merchant])
    prediction = model.predict(X_example)[0] 
    confidence=model.predict_proba(X_example).max()*100
    print(f"{merchant} -> {prediction} ({confidence:.0f}% confident)")

# Add at the end of train_model.py
print("\n Saving the model...")
with open('category_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)
print("Model saved as 'category_model.pkl'")