"""
Custom model classes for joblib serialization.
This module contains custom machine learning classes that need to be
importable at module level (not __main__) for proper joblib deserialization.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import logging

logger = logging.getLogger(__name__)


class HybridRFSVM(BaseEstimator, ClassifierMixin):
    """
    Hybrid classifier combining Random Forest and Support Vector Machine.
    
    This class combines predictions from Random Forest and SVM using a weighted
    voting mechanism with dynamic thresholds based on class distribution.
    
    Parameters
    ----------
    random_state : int, default=42
        Random state for reproducibility
    calibration_method : str, default='isotonic'
        Calibration method for SVM probability estimates
    use_smote : bool, default=True
        Whether to use SMOTE (currently not actively used in fit)
    """
    
    def __init__(self, random_state=42, calibration_method='isotonic', use_smote=True):
        self.random_state = random_state
        self.calibration_method = calibration_method
        self.use_smote = use_smote
        self.rf_model = None
        self.svm_model = None
        self.svm_calibrator = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        """
        Fit the hybrid model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels
            
        Returns
        -------
        self : HybridRFSVM
            Fitted estimator
        """
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        # Compute class weights for balanced training
        classes = np.unique(y_encoded)
        weights = compute_class_weight('balanced', classes=classes, y=y_encoded)
        class_weights = dict(zip(range(len(classes)), weights))
        
        X_train_final = X
        y_train_final = y_encoded
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            class_weight=class_weights,
            n_jobs=-1,
            bootstrap=True,
            max_features='sqrt'
        )
        self.rf_model.fit(X_train_final, y_train_final)
        self.feature_importances_ = self.rf_model.feature_importances_
        
        # Train calibrated SVM
        svm_base = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=False,
            random_state=self.random_state,
            class_weight=class_weights,
            decision_function_shape='ovr'
        )
        
        self.svm_calibrator = CalibratedClassifierCV(
            svm_base,
            method=self.calibration_method,
            cv=3
        )
        self.svm_calibrator.fit(X_train_final, y_train_final)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        # Get predictions and probabilities
        rf_pred = self.rf_model.predict(X)
        svm_proba = self.svm_calibrator.predict_proba(X)
        svm_pred = np.argmax(svm_proba, axis=1)
        svm_confidence = np.max(svm_proba, axis=1)
        
        # Compute dynamic weights based on class distribution
        class_counts = np.bincount(rf_pred)
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.max()
        
        # Hybrid decision with dynamic threshold
        hybrid_pred = []
        for i in range(len(rf_pred)):
            class_weight = class_weights[rf_pred[i]]
            threshold = 0.6 + (0.3 * class_weight)
            
            if svm_confidence[i] > threshold:
                hybrid_pred.append(svm_pred[i])
            else:
                hybrid_pred.append(rf_pred[i])
        
        hybrid_pred = np.array(hybrid_pred)
        return self.label_encoder.inverse_transform(hybrid_pred)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities (RF: 60%, SVM: 40%)
        """
        # Weighted average of RF and SVM probabilities
        rf_proba = self.rf_model.predict_proba(X)
        svm_proba = self.svm_calibrator.predict_proba(X)
        avg_proba = (0.6 * rf_proba) + (0.4 * svm_proba)
        return avg_proba / avg_proba.sum(axis=1, keepdims=True)
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
        y : array-like of shape (n_samples,)
            True labels
            
        Returns
        -------
        score : float
            Accuracy score
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'random_state': self.random_state,
            'calibration_method': self.calibration_method,
            'use_smote': self.use_smote
        }
    
    def set_params(self, **parameters):
        """Set parameters for this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
