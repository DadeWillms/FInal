from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd

df = pd.read_csv("cleaned_file.csv")


x = df[['StudyHours','SleepHours','AttendancePercent','AssignmentScore','ScreenTimeHours','ExerciseHours','SocialMediaHours','ProjectsCompleted','PartTimeJobHours']]
y = df['ExamScore']


model = DecisionTreeClassifier()
model.fit(x, y)

new_data = pd.DataFrame({
    'StudyHours':[5],
    'SleepHours':[100],
    'AttendancePercent':[2],
    'AssignmentScore':[1],
    'ScreenTimeHours':[0],
    'ExerciseHours':[5],
    'SocialMediaHours':[100],
    'ProjectsCompleted':[2],
    'PartTimeJobHours':[1],
})

prediction = model.predict(new_data)

importances = pd.Series(model.feature_importances_, index=x.columns)
print("Feature Importances:\n",importances.sort_values(ascending=False))

