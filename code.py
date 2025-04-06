

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------- #
#                           WORKFLOW CLASS                         #
# ---------------------------------------------------------------- #

class MLWorkflow:
    def __init__(self, dataset_path, target_column):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.df = None
        self.X = None
        self.y = None
        self.models = {}
        self.results = {}

    def load_data(self):
        logging.info("Loading dataset...")
        self.df = pd.read_csv(self.dataset_path)
        logging.info(f"Data shape: {self.df.shape}")
        return self.df

    def preprocess(self):
        logging.info("Preprocessing dataset...")
        self.df.dropna(inplace=True)
        for col in self.df.select_dtypes(include=['object']).columns:
            if col != self.target_column:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]
        logging.info("Preprocessing complete.")

    def eda(self):
        logging.info("Performing EDA...")
        print("First 5 rows:\n", self.df.head())
        print("\nSummary stats:\n", self.df.describe())
        print("\nClass balance:\n", self.df[self.target_column].value_counts())

        # Histograms
        self.df.hist(figsize=(12, 10), bins=20)
        plt.suptitle("Feature Distributions", fontsize=16)
        plt.show()

        # Correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Feature Correlation Heatmap")
        plt.show()

    def feature_engineering(self):
        logging.info("Performing feature engineering...")
        if self.X.shape[1] >= 2:
            self.X['interaction'] = self.X.iloc[:, 0] * self.X.iloc[:, 1]
        logging.info("Feature engineering done.")

    def split_data(self, test_size=0.2):
        logging.info("Splitting dataset...")
        return train_test_split(self.X, self.y, test_size=test_size, random_state=42)

    def define_models(self):
        logging.info("Defining models...")
        self.models = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForest': RandomForestClassifier(),
            'GradientBoosting': GradientBoostingClassifier(),
            'SVC': SVC(probability=True),
            'GaussianNB': GaussianNB(),
            'DecisionTree': DecisionTreeClassifier(),
            'KNeighbors': KNeighborsClassifier()
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        logging.info("Training and evaluating models...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]) if hasattr(model, "predict_proba") else None
            self.results[name] = {'model': model, 'accuracy': acc, 'auc': auc}

            logging.info(f"{name} Accuracy: {acc:.4f}, AUC: {auc:.4f}" if auc else f"{name} Accuracy: {acc:.4f}")
            print(f"\n{name} Classification Report:")
            print(classification_report(y_test, y_pred))
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    def hyperparameter_tuning(self, X_train, y_train):
        logging.info("Hyperparameter tuning for RandomForest...")
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
        }
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        logging.info(f"Best Params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def save_best_model(self):
        best = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        joblib.dump(best[1]['model'], f"{best[0]}_best_model.pkl")
        logging.info(f"Saved best model: {best[0]} with accuracy {best[1]['accuracy']:.4f}")

# ---------------------------------------------------------------- #
#                       SYNTHETIC DEMO DATA                        #
# ---------------------------------------------------------------- #

def generate_and_run_workflow():
    logging.info("Generating synthetic dataset...")
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
    columns = [f'feature_{i}' for i in range(10)]
    df = pd.DataFrame(X, columns=columns)
    df['target'] = y
    csv_path = 'synthetic_data.csv'
    df.to_csv(csv_path, index=False)

    # Run full workflow
    workflow = MLWorkflow(csv_path, target_column='target')
    workflow.load_data()
    workflow.preprocess()
    workflow.eda()
    workflow.feature_engineering()
    X_train, X_test, y_train, y_test = workflow.split_data()
    workflow.define_models()
    workflow.train_and_evaluate(X_train, X_test, y_train, y_test)
    workflow.hyperparameter_tuning(X_train, y_train)
    workflow.save_best_model()

# ---------------------------------------------------------------- #
#                               MAIN                               #
# ---------------------------------------------------------------- #

if __name__ == "__main__":
    generate_and_run_workflow()
