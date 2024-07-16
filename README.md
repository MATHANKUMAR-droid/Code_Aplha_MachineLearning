# Code_Aplha_MachineLearning
**#TASK 1 CREDI CARD ANALYSIS **
__________________________________________________________________________________________________________________________________________
import pandas as pd
data = pd.read_csv('credit_data.csv')
data.head()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Handle missing values (if any)
data = data.dropna()
# Split the data into features and target
X = data.drop('creditworthy', axis=1)
y = data['creditworthy']
# Define categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
# Preprocessing pipeline for numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# Initialize the models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'Gradient Boosting': GradientBoostingClassifier()
}
# Create a pipeline that includes preprocessing and model training
pipelines = {name: Pipeline(steps=[('preprocessor', preprocessor), ('model', model)]) for name, model in models.items()}
# Train and evaluate each model
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f'{name}:\n Accuracy: {accuracy:.4f}\n Precision: {precision:.4f}\n Recall: {recall:.4f}\n F1 Score: {f1:.4f}\n AUC-ROC: {auc:.4f}\n')
from sklearn.model_selection import GridSearchCV
# Example for tuning Random Forest
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(pipelines['Random Forest'], param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f'Best parameters: {grid_search.best_params_}')
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='roc_auc')
print(f'Cross-Validation AUC-ROC Scores: {cv_scores}')
print(f'Mean AUC-ROC Score: {cv_scores.mean():.4f}')
import joblib
joblib.dump(best_model, 'credit_scoring_model.pkl')
