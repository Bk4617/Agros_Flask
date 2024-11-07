import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
# Function to train and save the crop recommendation model
def train_crop_model(model_save_path):
    try:
        # Load the dataset
        file_path = 'C:\\Users\\ramakrishna\\OneDrive\\Desktop\\Agros_Flask\\Crop_recommendation.csv'
        df = pd.read_csv(file_path)
        print(f"Crop dataset shape: {df.shape}")  # Debugging line

        # Split features and labels
        X = df.drop('Crop', axis=1)
        y = df['Crop']

        # Encode the labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Save the label encoder for future use in decoding predictions
        joblib.dump(label_encoder, os.path.join(model_save_path, 'label_encoder.pkl'))

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Define base models
        base_models = [
            ('decision_tree', DecisionTreeClassifier(random_state=42)),
            ('random_forest', RandomForestClassifier(random_state=42)),  # No space here
            ('svm', SVC(probability=True, random_state=42)),
            ('knn', KNeighborsClassifier())
        ]

        # Define the meta-model
        meta_model = LogisticRegression(max_iter=1000, random_state=42)

        # Initialize Stacking Classifier
        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )

        # Define hyperparameters to tune for base models
        param_grid = {
            'decision_tree__max_depth': [10],
            'random_forest__n_estimators': [20],
            'random_forest__max_depth': [10],
            'svm__C': [0.1],
            'svm__kernel': ['linear'],
            'knn__n_neighbors': [5],
            'knn__weights': ['uniform']
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(stacking_model, param_grid, cv=4, n_jobs=-1, verbose=1)

        # Train the model with hyperparameter tuning
        grid_search.fit(X_train, y_train)

        # Save the best crop model
        joblib.dump(grid_search.best_estimator_, os.path.join(model_save_path, 'crop_model.pkl'))
        print("Crop model saved successfully.")

    except Exception as e:
        print(f"Error during crop model training or saving: {e}")

# Function to train and save the fertilizer model
def train_fertilizer_model(model_save_path):
    try:
        # Load the dataset
        file_path = 'C:\\Users\\ramakrishna\\OneDrive\\Desktop\\Agros_Flask\\f2.csv'
        df_fertilizer = pd.read_csv(file_path)
        print(f"Fertilizer dataset shape: {df_fertilizer.shape}")  # Debugging line

        # Split features and labels
        X = df_fertilizer[['Moisture', 'Soil_Type', 'Nitrogen', 'Potassium', 'Phosphorus']]
        y = df_fertilizer['Fertilizer']

        # Preprocessing
        numerical_features = ['Moisture', 'Nitrogen', 'Potassium', 'Phosphorus']
        categorical_features = ['Soil_Type']

        # Preprocessing for numerical data (Standardization + Imputation)
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Preprocessing for categorical data (One-Hot Encoding)
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Define base models
        base_models = [
            ('decision_tree', DecisionTreeClassifier(random_state=42)),
            ('random_forest', RandomForestClassifier(random_state=42)),  # No space here
            ('svm', SVC(probability=True, random_state=42)),
            ('knn', KNeighborsClassifier())
        ]

        # Define the meta-model
        meta_model = LogisticRegression(max_iter=1000, random_state=42)

        # Initialize Stacking Classifier
        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )

        # Create the full pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('stacking', stacking_model)])

        # Define hyperparameters to tune for base models
        param_grid = {
            'stacking__decision_tree__max_depth': [10],
            'stacking__random_forest__n_estimators': [20],
            'stacking__random_forest__max_depth': [10],
            'stacking__svm__C': [0.1],
            'stacking__svm__kernel': ['linear'],
            'stacking__knn__n_neighbors': [5],
            'stacking__knn__weights': ['uniform']
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(pipeline, param_grid, cv=4, n_jobs=-1, verbose=1)

        # Train the model with hyperparameter tuning
        grid_search.fit(X, y)

        # Save the best fertilizer model
        joblib.dump(grid_search.best_estimator_, os.path.join(model_save_path, 'fertilizer_model.pkl'))
        print("Fertilizer model saved successfully.")

    except Exception as e:
        print(f"Error during fertilizer model training or saving: {e}")

# Define the model save path
model_save_path = 'C:\\Users\\ramakrishna\\OneDrive\\Desktop\\Agros_Flask'

# Train and save the crop recommendation model
train_crop_model(model_save_path)

# Train and save the fertilizer model
train_fertilizer_model(model_save_path)