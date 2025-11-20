import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, classification_report, confusion_matrix)
from sklearn.metrics import mean_squared_error, r2_score

# Preparation 
df = pd.read_csv("merged.csv")
print(f"Total Respondants: {len(df)}\n")

df["Percentage"] = (df["Total Score"] / df["Max Points"]) * 100

def pass_or_fail(percentage):
    if percentage >= 70:
        return 1
    else:
        return 0
    
df["Pass/Fail"] = df["Percentage"].apply(pass_or_fail)

def get_letter_grade(percentage):
    if percentage >= 90:
        return 'A'
    elif percentage >= 80:
        return 'B'
    elif percentage >= 70:
        return 'C'
    elif percentage >= 60:
        return 'D'
    else:
        return 'F'
    
df["Letter Grade"] = df["Percentage"].apply(get_letter_grade)

numeric_features = [
    'About how long, in hours, did you study for exam 1?',
    'How many hours a day on average do you spend on sites with infinite scroll?',
    'How many hours of sleep did you get the night before the exam?'
]

categorical_features = [
    'I come to lecture:',
    "I've had prior machine learning / data science experience",
    'Did you do the readings?',
    'Did you leave the exam early?',
    'What year are you?'
]

X_num = df[numeric_features].copy()
X_num.columns = ['Study_Hours', 'Scroll_Hours', 'Sleep_Hours']
X_num = X_num.fillna(X_num.median())
X_cat = pd.get_dummies(df[categorical_features], drop_first=True, prefix=['Attendance', 'Experience', 'Readings', 'Left_Early', 'Year'])
X = pd.concat([X_num, X_cat], axis=1)

# Part 1
print("\nPASS / FAIL: \n")
y_pass_fail = df["Pass/Fail"].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y_pass_fail, test_size=0.4, random_state=27, stratify=y_pass_fail)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results_part1 = {}

print("_____ K-Nearest Neighbors _____")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_knn_predict = knn.predict(X_test_scaled)
print(f"Accuracy:  {accuracy_score(y_test, y_knn_predict):.4f}")
print(f"Precision: {precision_score(y_test, y_knn_predict):.4f}")
print(f"Recall:    {recall_score(y_test, y_knn_predict):.4f}")
results_part1['KNN'] = {
    'accuracy': accuracy_score(y_test, y_knn_predict),
    'predictions': y_knn_predict
}

print("\n_____ Logistic Regression _____")
logr = LogisticRegression(max_iter=1000, random_state=27)
logr.fit(X_train_scaled, y_train)
y_logr_predict = logr.predict(X_test_scaled)
print(f"Accuracy:  {accuracy_score(y_test, y_logr_predict):.4f}")
print(f"Precision: {precision_score(y_test, y_logr_predict):.4f}")
print(f"Recall:    {recall_score(y_test, y_logr_predict):.4f}")
results_part1['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, y_logr_predict),
    'predictions': y_logr_predict
}

print("\n_____ Decision Tree _____")
dt = DecisionTreeClassifier(max_depth=5, random_state=27)
dt.fit(X_train, y_train)
y_dt_predict = dt.predict(X_test)
print(f"Accuracy:  {accuracy_score(y_test, y_dt_predict):.4f}")
print(f"Precision: {precision_score(y_test, y_dt_predict):.4f}")
print(f"Recall:    {recall_score(y_test, y_dt_predict):.4f}")
results_part1['Decision Tree'] = {
    'accuracy': accuracy_score(y_test, y_dt_predict),
    'predictions': y_dt_predict
}


print("\nConfusion Matrices (Pass/Fail): ")


for model_name, result in results_part1.items():
    cm = confusion_matrix(y_test, result['predictions'])
    print(f"\n{model_name}:")
    print(cm)


# Part 2
print("\nLETTER GRADE: \n")
y_letter_grade = df['Letter Grade'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y_letter_grade, test_size=0.15, random_state=27, stratify=y_letter_grade)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("_____ K-Nearest Neighbors _____")
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)
y_knn_predict = knn.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_knn_predict):.4f}")
print(classification_report(y_test, y_knn_predict, zero_division=0))

print("\n_____ Logistic Regression _____")
logr = LogisticRegression(max_iter=1000, random_state=27)
logr.fit(X_train_scaled, y_train)
y_logr_predict = logr.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_logr_predict):.4f}")
print(classification_report(y_test, y_logr_predict, zero_division=0))

print("\n_____ Random Forest _____")
rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=27)
rf.fit(X_train, y_train)
y_rf_predict = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_rf_predict):.4f}")
print(classification_report(y_test, y_rf_predict, zero_division=0))


# Part 3
print("\nExact GRADE: \n")
y_score = df['Total Score'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y_score, test_size=0.25, random_state=27)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("_____ Linear Regression _____")
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_lin_predict = lin_reg.predict(X_test_scaled)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_lin_predict))}")
print(f"R² Score: {r2_score(y_test, y_lin_predict)}")

print("\n_____ Ridge Regression _____")
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train_scaled, y_train)
y_ridge_predict = ridge_reg.predict(X_test_scaled)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_ridge_predict))}")
print(f"R² Score: {r2_score(y_test, y_ridge_predict)}")


numeric_features = {
    "Study_Hours": "About how long, in hours, did you study for exam 1?",
    "Scroll_Hours": "How many hours a day on average do you spend on sites with infinite scroll?",
    "Sleep_Hours": "How many hours of sleep did you get the night before the exam?"
}
num_df = df[list(numeric_features.values()) + ["Percentage"]].copy()
num_df.columns = list(numeric_features.keys()) + ["Percentage"]
correlation = num_df.corr()["Percentage"].drop("Percentage").sort_values(ascending=False)

print("\nCorrelation of each numeric feature with grade :\n")
print(correlation)

