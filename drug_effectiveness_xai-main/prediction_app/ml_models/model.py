import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import FeatureHasher
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import shap
import matplotlib.pyplot as plt
import io
import base64
from django.conf import settings
import os

class DrugEffectivenessModel:
    def __init__(self):
        self.pipeline = None
        self.explainer = None
        self.preprocessor = None
        self.model = None
        
        # Download nltk data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        # Path to save models
        self.model_dir = os.path.dirname(os.path.abspath(__file__))
        self.pipeline_path = os.path.join(self.model_dir, 'pipeline.pkl')
        self.explainer_path = os.path.join(self.model_dir, 'explainer.pkl')
        
        # Load or train the model
        if os.path.exists(self.pipeline_path) and os.path.exists(self.explainer_path):
            self.load_model()
        else:
            print("Models not found. Please train the model first with the train() method.")
    
    def train(self, dataset_path):
        """Train the drug effectiveness model and save it for future use"""
        # 1. Load dataset
        df = pd.read_csv(dataset_path)
        
        # 2. Preprocessing
        # 2.1 Filter and standardize age categories
        allowed_ages = ['0-2','3-12','13-18','19-24','25-34','35-44','45-54','55-64','75 or over']
        df = df[df['Age'].isin(allowed_ages)].copy()
        df['age_group'] = df['Age'].astype(str)
        
        # 2.2 Encode gender
        df['Sex'] = df['Sex'].fillna('Unknown')
        le_gender = LabelEncoder()
        df['gender_enc'] = le_gender.fit_transform(df['Sex'])
        
        # 2.3 Hash-encode drugs to reduce dimensionality
        df['drug_list'] = df['Drug'].fillna('').apply(lambda x: [d.strip() for d in x.split(',') if d.strip()])
        hasher_drug = FeatureHasher(n_features=32, input_type='string')
        drug_hashed = hasher_drug.transform(df['drug_list']).toarray()
        drug_hash_cols = [f'drug_hash_{i}' for i in range(drug_hashed.shape[1])]
        df[drug_hash_cols] = drug_hashed
        
        # 2.4 Hash-encode symptoms similarly
        df['symptom_list'] = df['symptoms'].fillna('').apply(lambda x: [s.strip() for s in x.split(',') if s.strip()])
        hasher_sym = FeatureHasher(n_features=32, input_type='string')
        sym_hashed = hasher_sym.transform(df['symptom_list']).toarray()
        sym_hash_cols = [f'symptom_hash_{i}' for i in range(sym_hashed.shape[1])]
        df[sym_hash_cols] = sym_hashed
        
        # 2.5 Compute sentiment from reviews for target weighting
        sia = SentimentIntensityAnalyzer()
        df['review_sentiment'] = df['Reviews'].fillna('').apply(lambda x: sia.polarity_scores(str(x))['compound'])
        
        # 3. Compute weighted overallEffectiveness score
        df['overallEffectiveness'] = (
            df['review_sentiment'] * 0.4 +
            df['Satisfaction'] * 0.3 +
            df['Effectiveness'] * 0.2 +
            df['EaseofUse'] * 0.1
        )
        
        # 4. Prepare features and target
        target = 'overallEffectiveness'
        cat_feats = ['age_group', 'Condition']
        numeric_feats = ['gender_enc'] + drug_hash_cols + sym_hash_cols
        X = df[cat_feats + numeric_feats]
        y = df[target]
        
        # 5. Preprocessing pipeline for categorical features
        self.preprocessor = ColumnTransformer([
            ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_feats)
        ], remainder='passthrough')
        
        # 6. XGBoost regressor
        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            tree_method='hist',
            random_state=42
        )
        
        self.pipeline = Pipeline([
            ('pre', self.preprocessor),
            ('xgb', self.model)
        ])
        
        # 7. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 8. Train model
        self.pipeline.fit(X_train, y_train)
        
        # 9. Evaluate performance
        preds = self.pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        # 10. Create explainer
        X_train_trans = self.preprocessor.transform(X_train)
        self.explainer = shap.TreeExplainer(self.model)
        
        # 11. Save the pipeline and explainer
        self.save_model()
        
        return rmse, r2
    
    def save_model(self):
        """Save the trained model and explainer"""
        with open(self.pipeline_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        with open(self.explainer_path, 'wb') as f:
            pickle.dump(self.explainer, f)
    
    def load_model(self):
        """Load the trained model and explainer"""
        with open(self.pipeline_path, 'rb') as f:
            self.pipeline = pickle.load(f)
            # Extract the preprocessor and model
            self.preprocessor = self.pipeline.named_steps['pre']
            self.model = self.pipeline.named_steps['xgb']
        
        with open(self.explainer_path, 'rb') as f:
            self.explainer = pickle.load(f)
    
    def preprocess_input(self, age_group, sex, condition, drugs, symptoms):
        """Preprocess a single input for prediction"""
        # Create a DataFrame with the input
        input_df = pd.DataFrame({
            'age_group': [age_group],
            'Condition': [condition],
            'gender_enc': [0 if sex == 'Male' else 1 if sex == 'Female' else 2]  # Simple encoding
        })
        
        # Hash-encode drugs
        drug_list = [drugs.split(',')]
        hasher_drug = FeatureHasher(n_features=32, input_type='string')
        drug_hashed = hasher_drug.transform(drug_list).toarray()
        drug_hash_cols = [f'drug_hash_{i}' for i in range(drug_hashed.shape[1])]
        for i, col in enumerate(drug_hash_cols):
            input_df[col] = drug_hashed[0][i]
        
        # Hash-encode symptoms
        symptom_list = [symptoms.split(',')]
        hasher_sym = FeatureHasher(n_features=32, input_type='string')
        sym_hashed = hasher_sym.transform(symptom_list).toarray()
        sym_hash_cols = [f'symptom_hash_{i}' for i in range(sym_hashed.shape[1])]
        for i, col in enumerate(sym_hash_cols):
            input_df[col] = sym_hashed[0][i]
        
        return input_df
    
    def predict(self, age_group, sex, condition, drugs, symptoms):
        """Make a prediction for a single input"""
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        input_df = self.preprocess_input(age_group, sex, condition, drugs, symptoms)
        effectiveness = self.pipeline.predict(input_df)[0]
        
        return effectiveness
    
    def explain_prediction(self, age_group, sex, condition, drugs, symptoms):
        """Generate SHAP explanation for a prediction"""
        if self.pipeline is None or self.explainer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        input_df = self.preprocess_input(age_group, sex, condition, drugs, symptoms)
        input_transformed = self.preprocessor.transform(input_df)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(input_transformed)
        
        # Generate and save the SHAP plot
        plt.figure(figsize=(10, 6))
        feature_names = self.preprocessor.get_feature_names_out()
        shap.force_plot(
            self.explainer.expected_value, 
            shap_values[0], 
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Convert to base64 string
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return image_base64
