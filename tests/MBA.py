import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
from collections import Counter

columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
           "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]

df = pd.read_csv("raw.csv", names=columns)

for col in columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  

df = df.fillna(0)


def categorize(row):
    items = []
    items.append("High_Glucose" if row["Glucose"] > 140 else "Normal_Glucose")
    items.append("High_BMI" if row["BMI"] > 30 else "Normal_BMI")
    items.append("Older" if row["Age"] > 40 else "Younger")
    items.append("Diabetes" if row["Outcome"] == 1 else "No_Diabetes")
    return items

transactions = df.apply(categorize, axis=1)

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
basket_df = pd.DataFrame(te_array, columns=te.columns_)


min_support = 0.4
frequent_items = apriori(basket_df, min_support=min_support, use_colnames=True)


min_confidence = 0.7
rules = association_rules(frequent_items, metric="confidence", min_threshold=min_confidence)


all_items = [item for transaction in transactions for item in transaction]

item_counts = Counter(all_items)

item_counts = dict(sorted(item_counts.items(), key=lambda x: x[1], reverse=True))



rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ', '.join(list(x)))
rules["consequents_str"] = rules["consequents"].apply(lambda x: ', '.join(list(x)))
rules["rule"] = rules["antecedents_str"] + " -> " + rules["consequents_str"]

rules_sorted = rules.sort_values(by="confidence", ascending=False)

plt.figure(figsize=(10,6))
plt.barh(rules_sorted["rule"], rules_sorted["confidence"], color='skyblue')
plt.xlabel("Confidence")
plt.ylabel("Rules")
plt.title("Association Rules by Confidence")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
plt.scatter(rules["support"], rules["confidence"], color='orange')
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")

for i, txt in enumerate(rules["rule"]):
    plt.annotate(txt, (rules["support"].iloc[i], rules["confidence"].iloc[i]), fontsize=8)

plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.bar(item_counts.keys(), item_counts.values(), color='teal')
plt.xlabel("Feature")
plt.ylabel("Frequency")
plt.title("Feature Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])