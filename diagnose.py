import pickle

with open('category_model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)

test_merchants = ["Lowe's", "Home Depot", "Menards", "Ace Hardware", "Walmart"]

for merchant in test_merchants:
    X = vectorizer.transform([merchant])
    probs = model.predict_proba(X)[0]
    classes = model.classes_
    ranked = sorted(zip(classes, probs), key=lambda x: -x[1])
    print(f"\n{merchant}:")
    for cls, p in ranked[:3]:
        print(f"  {cls}: {p*100:.1f}%")