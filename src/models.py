"""
Models Module for Diabetes Classification.
Defines Conventional ML Baseline Classifiers and Multilayer Perceptron (MLP) architectures.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.optim as optim

def get_baseline_models(random_state=42):
    """
    Returns dictionary of baseline machine learning classifiers.
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

class PyTorchMLP(nn.Module):
    """
    PyTorch implementation of Multilayer Perceptron for Diabetes Prediction.
    Architecture:
      Input (D) -> Dense(hidden1) -> BatchNorm -> ReLU -> Dropout(p)
               -> Dense(hidden2) -> BatchNorm -> ReLU -> Dropout(p)
               -> Dense(1) -> Sigmoid -> Prediction
    """
    def __init__(self, input_dim, hidden1=64, hidden2=32, dropout_rate=0.2):
        super(PyTorchMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)
