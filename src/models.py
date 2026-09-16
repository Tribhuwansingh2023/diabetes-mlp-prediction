"""
Models Module for Diabetes Classification.
Defines Conventional Machine Learning Baseline Classifiers and the
authoritative Multilayer Perceptron (MLP) architecture using Scikit-Learn.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def get_baseline_models(random_state=42):
    """
    Returns a dictionary of baseline machine learning classifiers.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=random_state
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=7, weights='distance'
        ),
        'Support Vector Machine': SVC(
            C=1.0, kernel='rbf', probability=True, random_state=random_state
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5, min_samples_split=10, random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150, max_depth=6, random_state=random_state
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=random_state
        ),
        'MLP (Scikit-Learn Baseline)': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.005,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=random_state
        )
    }
    return models

def build_mlp_classifier(hidden_layer_sizes=(64, 32), activation='relu',
                         learning_rate_init=0.001, alpha=0.001,
                         batch_size=32, max_iter=500, early_stopping=False,
                         random_state=42):
    """
    Factory function to construct a configured Scikit-Learn MLPClassifier.
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver='adam',
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        random_state=random_state
    )
