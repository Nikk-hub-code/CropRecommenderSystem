TEXT-BASED FLOW DESCRIPTION

    START
    │
    ├── IMPORT LIBRARIES
    │   (pandas, numpy, matplotlib, seaborn, sklearn, xgboost)
    │
    ├── LOAD DATASET
    │   ("Crop_recommendation.csv")
    │   ↓
    ├── EXPLORE DATA
    │   ├── Print dataset shape
    │   ├── Show first 5 rows
    │   ├── Display dataset info
    │   └── Show label distribution
    │
    ├── DATA PREPROCESSING
    │   ├── Check for missing values
    │   ├── Drop null values (if any)
    │   ├── Encode categorical labels (LabelEncoder)
    │   ├── Separate features (X) and target (Y)
    │   ├── Split data into train/test sets (80/20)
    │   └── Scale features (StandardScaler)
    │
    ├── MODEL TRAINING
    │   │
    │   ├── RANDOM FOREST
    │   │   ├── Define parameter grid
    │   │   ├── Perform GridSearchCV (5-fold)
    │   │   ├── Find best parameters
    │   │   └── Store best model
    │   │
    │   ├── XGBOOST
    │   │   ├── Define parameter grid
    │   │   ├── Perform GridSearchCV (5-fold)
    │   │   ├── Find best parameters
    │   │   └── Store best model
    │   │
    │   └── ENSEMBLE MODEL
    │       ├── Create VotingClassifier
    │       │   (Random Forest + XGBoost, soft voting)
    │       └── Train ensemble model
    │
    ├── MODEL EVALUATION
    │   ├── Define evaluation function that:
    │   │   ├── Calculates accuracy
    │   │   ├── Generates classification report
    │   │   └── Plots confusion matrix
    │   │
    │   ├── Evaluate all three models:
    │   │   ├── Random Forest
    │   │   ├── XGBoost
    │   │   └── Ensemble
    │   │
    │   └── Compare model accuracies
    │
    ├── FEATURE IMPORTANCE ANALYSIS
    │   ├── Extract feature importances from both models
    │   ├── Create comparison DataFrame
    │   ├── Plot feature importance chart
    │   └── Display ranked feature importance
    │
    ├── CREATE PREDICTION FUNCTION
    │   (predict_crop_recommendation)
    │   ├── Scale input features
    │   ├── Make prediction
    │   ├── Get crop name from label encoder
    │   ├── Calculate confidence score
    │   └── Return top 3 recommendations
    │
    └── USER INTERACTION
        ├── Display program header
        ├── Prompt user for input parameters:
        │   (N, P, K, temperature, humidity, ph, rainfall)
        ├── Call prediction function
        ├── Display results:
        │   ├── Recommended crop
        │   ├── Confidence level
        │   └── Top 3 recommendations
        │
    └── Error handling for invalid input
    │
    END
