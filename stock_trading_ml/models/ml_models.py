"""
ML model implementations for trading strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow, handle gracefully if not available
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    print("✅ TensorFlow imported successfully")
except ImportError:
    TF_AVAILABLE = False
    tf = None
    print("Warning: TensorFlow not available. LSTM models will be disabled.")


class MLModelManager:
    """
    Manages multiple ML models for trading strategy.
    """
    
    def __init__(self, config: Dict):
        """Initialize model manager with configuration."""
        self.config = config
        self.models_config = config['models']
        self.random_state = self.models_config['random_state']
        
        # Model storage
        self.models = {}
        self.trained_models = {}
        self.performance_metrics = {}
        
        # Available models (filter out LSTM if TensorFlow not available)
        self.available_models = ['logistic_regression', 'random_forest', 'xgboost']
        if TF_AVAILABLE:
            self.available_models.append('lstm')
        
        print(f"Available models: {self.available_models}")
    
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all ML models."""
        print("Initializing ML models...")
        
        # Logistic Regression
        self.models['logistic_regression'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        # LSTM (only if TensorFlow is available)
        if TF_AVAILABLE:
            print("TensorFlow available - LSTM model will be created during training")
        else:
            print("TensorFlow not available - LSTM model disabled")
        
        return self.models
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train all available models."""
        print(f"Training models on {len(X)} samples with {len(X.columns)} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.models_config['test_size'], 
            random_state=self.random_state, stratify=y
        )
        
        models_to_train = [m for m in self.models_config['models_to_train'] 
                          if m in self.available_models]
        
        for model_name in models_to_train:
            print(f"\nTraining {model_name}...")
            
            try:
                if model_name == 'lstm' and TF_AVAILABLE:
                    trained_model = self._train_lstm(X_train, y_train, X_test, y_test)
                else:
                    trained_model = self._train_sklearn_model(
                        model_name, X_train, y_train, X_test, y_test
                    )
                
                self.trained_models[model_name] = trained_model
                print(f"✅ {model_name} trained successfully")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                continue
        
        return self.trained_models
    
    def _train_sklearn_model(self, model_name: str, X_train: pd.DataFrame, 
                           y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Any:
        """Train scikit-learn or XGBoost model."""
        model = self.models[model_name]
        
        # Get hyperparameters for grid search
        if model_name in self.models_config['hyperparameters']:
            param_grid = self.models_config['hyperparameters'][model_name]
            
            # Grid search
            grid_search = GridSearchCV(
                model, param_grid, 
                cv=self.models_config['cv_folds'],
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            
        else:
            # Train with default parameters
            best_model = model
            best_model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)
        
        try:
            train_prob = best_model.predict_proba(X_train)[:, 1]
            test_prob = best_model.predict_proba(X_test)[:, 1]
        except:
            train_prob = train_pred
            test_prob = test_pred
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, test_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, test_pred, average='weighted', zero_division=0),
        }
        
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, test_prob)
        except:
            metrics['roc_auc'] = 0.0
        
        self.performance_metrics[model_name] = metrics
        
        return best_model
    
    def _train_lstm(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series) -> Optional[Any]:
        """Train LSTM model (only if TensorFlow available)."""
        if not TF_AVAILABLE:
            print("Cannot train LSTM: TensorFlow not available")
            return None
            
        # Reshape data for LSTM (samples, timesteps, features)
        # For simplicity, use sequence length of 1
        X_train_lstm = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test_lstm = X_test.values.reshape(X_test.shape, 1, X_test.shape[1])
        
        # Create LSTM model
        model = Sequential([
            LSTM(64, return_sequences=False, input_shape=(1, X_train.shape[1])),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train_lstm, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test_lstm, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train_lstm, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test_lstm, y_test, verbose=0)
        
        self.performance_metrics['lstm'] = {
            'train_accuracy': train_acc,
            'val_accuracy': test_acc,
            'train_loss': train_loss,
            'val_loss': test_loss
        }
        
        return model
    
    def predict(self, model_name: str, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using specified model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.trained_models[model_name]
        
        if model_name == 'lstm' and TF_AVAILABLE:
            # Reshape for LSTM
            X_lstm = X.values.reshape(X.shape[0], 1, X.shape[1])
            probabilities = model.predict(X_lstm, verbose=0).flatten()
            predictions = (probabilities > 0.5).astype(int)
        else:
            # Scikit-learn or XGBoost model
            predictions = model.predict(X)
            try:
                probabilities = model.predict_proba(X)[:, 1]
            except:
                probabilities = predictions.astype(float)
        
        return predictions, probabilities
    
    def get_feature_importance(self, model_name: str) -> pd.DataFrame:
        """Get feature importance for tree-based models."""
        if model_name not in self.trained_models:
            return pd.DataFrame()
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            feature_names = self.models[model_name].feature_names_in_ if hasattr(model, 'feature_names_in_') else [f'feature_{i}' for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return pd.DataFrame()
    
    def save_models(self, save_dir: str) -> None:
        """Save trained models to disk."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            if model_name == 'lstm' and TF_AVAILABLE:
                model.save(f"{save_dir}/{model_name}_model.h5")
            else:
                joblib.dump(model, f"{save_dir}/{model_name}_model.pkl")
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir: str) -> None:
        """Load trained models from disk."""
        import os
        
        for model_name in self.available_models:
            if model_name == 'lstm' and TF_AVAILABLE:
                model_path = f"{save_dir}/{model_name}_model.h5"
                if os.path.exists(model_path):
                    self.trained_models[model_name] = tf.keras.models.load_model(model_path)
            else:
                model_path = f"{save_dir}/{model_name}_model.pkl"
                if os.path.exists(model_path):
                    self.trained_models[model_name] = joblib.load(model_path)
        
        print(f"Models loaded from {save_dir}")
