import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


label_name='AccountantDesc'
# Custom transformer for data preprocessing
class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_map):
        self.feature_map = feature_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Fill missing values with empty strings
        X = X.fillna('')
        # Combine relevant text fields based on feature_map
        X_combined = X[self.feature_map].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        return X_combined

def try_classify():
    # Load data
    full_data  = pd.read_csv('purchase_order.csv')  # Replace with your actual data file
    # Drop rows where the target label ('AccountantCode') is NaN
    full_data = full_data.dropna(subset=[label_name])

    feature_df = pd.read_excel('name-desc-map.xlsx')
    feature_map=feature_df[feature_df['type']=='train']['name']

    # Convert AccountantCode to categorical type
    full_data[label_name] = full_data[label_name].astype(str)

    # Data preprocessing
    # Assuming 'PURCHASE_DESC' and similar fields contain text, and 'AccountantCode' is the label
    feature_data = full_data[feature_map]  # Add other relevant text fields
    feature_data = feature_data.fillna('')
    labels = full_data[label_name]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.2, random_state=42)

    # Preprocessing and model pipeline
    pipeline = Pipeline([
        ('preprocessor', DataPreprocessor(feature_map=feature_map)),
        ('vectorizer', TfidfVectorizer(max_features=1000)),  # Adjust features based on your data
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Model training
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

    # # Prediction
    # y_pred = pipeline.predict(X_test)
    #
    # # Evaluation
    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("Classification Report:\n", classification_report(y_test, y_pred))
    # Extract the trained classifier from the pipeline
    classifier = pipeline.named_steps['classifier']

    # Get the feature importances from the classifier
    feature_importances = classifier.feature_importances_

    # Get the feature names from the vectorizer
    feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out()

    # Display feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })

    # Sort the features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print("Feature Importances:")
    print(feature_importance_df)


def main():
    try_classify()


if __name__ == '__main__':
    main()
