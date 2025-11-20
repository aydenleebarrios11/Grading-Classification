# Grading-Classification
This is a machine learning project where I used various algorithms to bin grades into pass / fail, letter grades, and used regression to try and predict exact grades

Introduction
In professor Morawski’s class students were asked to fill out a survey that would indicate their habits related to exam grade. The goal of this analysis was to understand how different study habits and behaviors relate to student performance, using a dataset containing survey responses and grade data from 188 students. Each record included both numerical and categorical data such as study hours, sleep before the exam, time spent on social media, class attendance, and whether students left the exam early. Three types of predictive models were built:
Predict whether a student passes or fails.


Predict the student’s letter grade A through F


Predict the student’s exact numerical score.


For each part, several machine learning algorithms were trained. The dataset was preprocessed by filling missing values, scaling numerical features, and one-hot encoding categorical variables. Models were evaluated using accuracy, precision, recall, and regression metrics with RMSE and R^2.
Feature Engineering
Each student’s performance was expressed as Total score / Max Points:
Grades were then binned into both binary Pass/Fail and letter grade categories:
Pass (1) for A, B, C (≥70%)


Fail (0) for D, F (<70%)


Features were divided into:
Numerical: hours studied, hours of social media use, and hours of sleep before the exam.


Categorical: attendance frequency, prior ML experience, completion of readings, early departure, and year (grade level)


Categorical variables were converted to binary indicators using pd.get_dummies() and numerical features were standardized with the StandardScalar function.
The final dataset contained 188 responses
Predicting Pass Vs Fail
Three models were trained: KNN, Logistic Regression, and Decision Tree
A 60/40 train test split was used, stratified by the pass/fail label. Evaluation metrics included accuracy, precision, and recall.
Model
Accuracy
Precision
Recall
K-Nearest Neighbors
0.66
0.58
0.58
Logistic Regression
0.71
0.76
0.42
Decision Tree
0.55
0.45
0.48

Analysis:
Logistic Regression performed best overall with 71% accuracy, indicating a reasonably strong linear relationship between student habits and passing outcomes. However, the recall of 0.42 shows that the model missed many failing students. It’s better at confirming who passed than detecting who might fail.

The KNN model had more balanced precision and recall, meaning it’s less confident but more even handed. The Decision Tree underperformed, possibly due to overfitting small subsets or the limited dataset size. Confusion matrices were printed to visualize classification balance. Logistic Regression showed very few false positives and a relatively balanced split between pass/fail predictions.

Students who studied more hours, slept adequately, and consistently attended lectures tended to fall into the “Pass” group. However, nonlinear interactions between habits along with other outliers likely limited the models’ predictive power.
Predicting Letter Grades A–F
For the second task, letter grades were predicted using KNN, Logistic Regression, and Random Forest
The data was split 85/15 train/test and evaluated using accuracy and the classification report (precision, recall).
Model


Accuracy
K-Nearest Neighbors


   0.38
Logistic Regression


   0.52
Random Forest


   0.41

Analysis:
All models struggled to differentiate between individual letter grades. The Logistic Regression model achieved the best performance at 52% accuracy. Most misclassifications occurred among the mid grade categories C and D, which had overlapping feature distributions. It was found that larger train sets for the data yielded better results.

Letter grades represent fine grained distinctions that are difficult to separate using behavioral data alone. Features like “study hours” and “sleep” may correlate weakly with small score differences, making it hard for models to predict specific letters. The dataset also has class imbalance such as very little A’s, which reduces model performance for underrepresented classes like A.
Predicting Exact Grade
Two regression models were tested:
Model
RMSE
 R^2
Linear Regression
8.50
0.18
Ridge Regression
8.49
0.18

Analysis:
Both regression models explained roughly 18% of the variance in total scores, suggesting that the selected study behavior features only modestly predict performance. The RMSE being 8.5 points indicates that the models’ grade predictions typically deviate by about one letter grade from the true score. Utilizing Ridge did not meaningfully improve results, confirming that overfitting was not the main issue; rather, the available features lacked strong predictive power to the level that would be required to predict exact scores.
Conclusions
Across all models, Logistic Regression consistently provided the best accuracy and interpretability for predicting whether a student passes or fails. For multiclass grading, models struggled due to overlapping categories and limited data size.
From a behavioral perspective, the analysis suggests:
Sleep duration are weak but positive predictors of success


Screen time has negligent correlation to grade


Study hours has a slight negative correlation to grade (likely due to outliers)


However, the modest performance metrics imply that grades depend on additional factors not captured in the survey (cmsc gpa ,  motivation, studied with friends?)
To improve predictive power we could incorporate more statistical and behavioral data (study schedules, GPA history) and analyze the importance of each category to identify which behaviors most affect grades. In addition, simply having more data would also assist in finding the best results.
Overall, this project demonstrates how machine learning can reveal patterns in student behavior, even if precise grade prediction remains challenging. Logistic Regression proved to be the most reliable model for broad outcomes pass/fail, while regression models were less effective for predicting exact scores.








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









Total Respondants: 188

PASS / FAIL: 

_____ K-Nearest Neighbors _____
Accuracy:  0.6579
Precision: 0.5806
Recall:    0.5806

_____ Logistic Regression _____
Accuracy:  0.7105
Precision: 0.7647
Recall:    0.4194

_____ Decision Tree _____
Accuracy:  0.5526
Precision: 0.4545
Recall:    0.4839

Confusion Matrices (Pass/Fail): 

KNN:
[[32 13]
 [13 18]]

Logistic Regression:
[[41  4]
 [18 13]]

Decision Tree:
[[27 18]
 [16 15]]

LETTER GRADE: 

_____ K-Nearest Neighbors _____
Accuracy: 0.3793
              precision    recall  f1-score   support

           A       0.00      0.00      0.00         1
           B       0.33      0.20      0.25         5
           C       0.20      0.17      0.18         6
           D       0.50      0.29      0.36         7
           F       0.41      0.70      0.52        10

    accuracy                           0.38        29
   macro avg       0.29      0.27      0.26        29
weighted avg       0.36      0.38      0.35        29


_____ Logistic Regression _____
Accuracy: 0.5172
              precision    recall  f1-score   support

           A       0.00      0.00      0.00         1
           B       0.33      0.20      0.25         5
           C       0.50      0.17      0.25         6
           D       0.50      0.43      0.46         7
           F       0.56      1.00      0.71        10

    accuracy                           0.52        29
   macro avg       0.38      0.36      0.34        29
weighted avg       0.47      0.52      0.45        29


_____ Random Forest _____
Accuracy: 0.4138
              precision    recall  f1-score   support

           A       0.00      0.00      0.00         1
           B       0.00      0.00      0.00         5
           C       0.25      0.17      0.20         6
           D       0.75      0.43      0.55         7
           F       0.40      0.80      0.53        10

    accuracy                           0.41        29
   macro avg       0.28      0.28      0.26        29
weighted avg       0.37      0.41      0.36        29


Exact GRADE: 

_____ Linear Regression _____
RMSE: 8.496208910654236
R² Score: 0.1797636572714021

_____ Ridge Regression _____
RMSE: 8.488994351538677
R² Score: 0.18115607385646182

Correlation of each numeric feature with grade :

Sleep_Hours     0.364455
Scroll_Hours   -0.001973
Study_Hours    -0.129932
Name: Percentage, dtype: float64
