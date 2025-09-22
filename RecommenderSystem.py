import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# **Load Dataset**
df = pd.read_csv("Crop_recommendation.csv")
print("Dataset Shape: ",df.shape)
print("\nFirst 5 Rows: ")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nLabel Distribution:")
print(df['label'].value_counts())


# **Data Preprocessing**
# Check for missing values
print("Missing Values:\n", df.isnull().sum()) 

#Handle Missing values if any
df = df.dropna() 

#Encode categorial labels
label_encoder = LabelEncoder() 
df['label_encoder'] = label_encoder.fit_transform(df['label'])

#Separate features and target
X = df.drop(['label', 'label_encoder'], axis=1) 
Y = df['label_encoder']

#Split the data
X_train, X_test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

#Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")


#**Random Forest Model**
#Random Forest with hyperparameter tuning
rf_parameter = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_parameter,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rf_grid.fit(X_train_scaled, Y_Train)
best_rf = rf_grid.best_estimator_

print("Best Random Forest Parameters: ", rf_grid.best_params_)
print("Random Forest Best CV Score: ", rf_grid.best_score_)


#**XGBoost Model**
#XGBoost with hyperparameter tuning
xgb_parameter = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}
xgb_grid = GridSearchCV(
    XGBClassifier(random_state = 42, use_label_encoder = False, eval_metric = 'mlogloss'),
    xgb_parameter,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
xgb_grid.fit(X_train_scaled, Y_Train)
best_xgb = xgb_grid.best_estimator_

print("Best XGBoost Parameters: ",xgb_grid.best_params_)
print("XGBoost Best CV Score: ",xgb_grid.best_score_)


#**Ensemble Model**
from sklearn.ensemble import VotingClassifier

#Create ensemble of both models
ensemble = VotingClassifier(
    estimators=[
        ('random forest', best_rf),
        ('xgboost', best_xgb)
    ],
    voting='soft' #use soft voting for probability-based predictions
)

#Train ensemble model
ensemble.fit(X_train_scaled, Y_Train)

#**Evaluate Models**
def evaluate_model(model, X_test, Y_Test, model_name):
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_Test, Y_pred)

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(Y_Test, Y_pred, target_names=label_encoder.classes_))

    #Confusion Matrix
    plt.figure(figsize=(10,8))
    cm = confusion_matrix(Y_Test, Y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    return accuracy

#Evaluate all models
rf_accuracy = evaluate_model(best_rf, X_test_scaled, Y_Test, "Random Forest")
xgb_accuracy = evaluate_model(best_xgb, X_test_scaled, Y_Test, "XGBoost")
ensemble_accuracy = evaluate_model(ensemble, X_test_scaled, Y_Test, "Ensemble(RF + XGBoost)")

#Compare accuracies
print("\nModel Comparison:")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")


#**Feature Importance Analysis
#Feature importance from Random Forest
rf_feature_importance = best_rf.feature_importances_
feature_names = X.columns

#create feature importance DataFrame
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'random_forest_importance': rf_feature_importance,
    'xgboost_importance': best_xgb.feature_importances_
})

#Plot feature importance
plt.figure(figsize=(12,8))
feature_importance_df.set_index('feature').plot(kind='bar', figsize=(12,8))
plt.title("Feature Importance Comparison")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Feature Importance Rankings:")
print(feature_importance_df.sort_values('random_forest_importance', ascending=False))


#**Create Prediction Function
def predict_crop_recommendation(model, input_feature, scaler, label_encoder):
    """
    Predict crop recommendation based on input features
    """
    #Convert input to numpy array and scale
    input_array = np.array(input_feature).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    #Predict
    prediction = model.predict(input_scaled)
    prediction_prob = model.predict_proba(input_scaled)

    #Get crop name and confidence
    crop_name = label_encoder.inverse_transform(prediction)[0]
    confidence = np.max(prediction_prob)

    #Get top 3 recommendation
    top_3_indices = np.argsort(prediction_prob[0])[-3:][::-1]
    top_3_crops = label_encoder.inverse_transform(top_3_indices)
    top_3_confidence = prediction_prob[0][top_3_indices]

    return {
        'recommended_crop': crop_name,
        'confidence': confidence,
        'top_3_recommendation': list(zip(top_3_crops, top_3_confidence))
    }


#**User Interaction**
print("\n"+"="*50)
print("\tCrop Recommending System")
print("="*50)
try:
    N = float(input("Nitrogen: "))
    P = float(input("Phosphorus: "))
    K = float(input("Potassium: "))
    temperature = float(input("Temperature: "))
    humidity = float(input("Humidity: "))
    ph = float(input("pH: "))
    rainfall = float(input("Rainfall: "))

    user_input = [N, P, K, temperature, humidity, ph, rainfall]

    result = predict_crop_recommendation(ensemble, user_input, scaler, label_encoder)

    print("\n"+"="*50)
    print("Crop Recommendation")
    print("="*50)
    print(f"Recommended Crop: {result['recommended_crop']} (Confidence: {result['confidence']:.2%})")
    print("\nTop 3 Recommendation:")
    for crop, confidence in result['top_3_recommendation']:
        print(f"{crop}: {confidence:.2f}")
except ValueError:
    print("Please enter valid numerical values")
