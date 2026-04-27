from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd

df = pd.read_csv("data.csv")


x = df[['login_attempts','session_duration','failed_logins','ip_reputation','unusual_time_login']]
y = df['suspicious']

model = DecisionTreeClassifier()
model.fit(x, y)

new_data = pd.DataFrame({
    'login_attempts':[5],
    'session_duration':[100],
    'failed_logins':[2],
    'ip_reputation':[1],
    'unusual_time_login':[0]
})

prediction = model.predict(new_data)

importances = pd.Series(model.feature_importances_, index=x.columns)
print("Feature Importances:\n",importances.sort_values(ascending=False))

rules = export_text(model, feature_names=list(x.columns))
print("Decision Tree Rules:\n",rules)
