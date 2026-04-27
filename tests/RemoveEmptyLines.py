import pandas as pd

df = pd.read_csv('midterm_knn.csv')

df_cleaned = df.dropna(subset=['StudyHours','SleepHours','AttendancePercent','AssignmentScore','ScreenTimeHours','ExerciseHours','SocialMediaHours','ProjectsCompleted','PartTimeJobHours','ExamScore'])

df_cleaned.to_csv('cleaned_file.csv', index=False)


