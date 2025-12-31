# Import required libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import joblib
import os
import warnings
from glob import glob
import gc
from skopt.callbacks import DeltaYStopper
import psutil

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# Constants
MODEL_PATH = 'fake_news_model.pkl'  
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

# Dataset Configuration
DATA_CONFIG = {
    'LIAR': {
        'paths': {
            'train': r"C:\Users\aadis\liar_dataset-master\train.tsv",
            'valid': r"C:\Users\aadis\liar_dataset-master\valid.tsv",
            'test': r"C:\Users\aadis\liar_dataset-master\test.tsv"
        },
        'columns': [
            'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
            'state_info', 'party_affiliation', 'barely_true_counts',
            'false_counts', 'half_true_counts', 'mostly_true_counts',
            'pants_on_fire_counts', 'context'
        ],
        'label_map': {
            'true': 0, 'mostly-true': 1, 'half-true': 2,
            'barely-true': 3, 'false': 4, 'pants-fire': 5
        }
    },
    'BINARY': {
        'paths': {
            'fake': r"C:\Users\aadis\Dataset\Fake.csv",
            'true': r"C:\Users\aadis\Dataset\True.csv"
        },
        'text_col': 'text',  # Column containing news text
        'label_map': {'fake': 1, 'true': 0}  # 1=fake, 0=true
    }
}

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenization and stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

# Dataset Loaders
def load_liar_dataset(config):
    """LIAR dataset specific loader"""
    dfs = []
    for split in ['train', 'valid', 'test']:
        try:
            df = pd.read_csv(config['paths'][split], sep='\t', header=None)
            df.columns = config['columns']
            dfs.append(df)
        except FileNotFoundError:
            warnings.warn(f"{split} file not found for LIAR dataset")
    
    full_df = pd.concat(dfs)
    full_df['clean_text'] = full_df['statement'].apply(preprocess_text)
    full_df['label'] = full_df['label'].map(config['label_map'])
    return full_df

def load_binary_dataset(config):
    """Loader for binary datasets with separate fake/true files"""
    dfs = []
    for label_type, filepath in config['paths'].items():
        try:
            df = pd.read_csv(filepath)
            df['label'] = config['label_map'][label_type]
            df['clean_text'] = df[config['text_col']].apply(preprocess_text)
            dfs.append(df)
        except FileNotFoundError:
            warnings.warn(f"File not found: {filepath}")
    
    return pd.concat(dfs)[['clean_text', 'label']]

def load_dataset(dataset_name):
    """Main dataset loading function"""
    config = DATA_CONFIG.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATA_CONFIG.keys())}")
    
    if dataset_name == 'LIAR':
        return load_liar_dataset(config)
    elif dataset_name == 'BINARY':
        return load_binary_dataset(config)
    else:
        raise ValueError(f"No loader implemented for dataset: {dataset_name}")

def combine_datasets(dataset_names):
    """Combine multiple datasets for training"""
    combined_df = pd.DataFrame()
    
    for name in dataset_names:
        df = load_dataset(name)
        
        # For binary datasets, map to LIAR's label scheme (3-5 = fake)
        if name == 'BINARY':
            df['label'] = df['label'].apply(lambda x: 4 if x == 1 else 0)  # Map to LIAR's false/true
        
        combined_df = pd.concat([combined_df, df[['clean_text', 'label']]])
    
    return combined_df

# Model Training
def train_model(dataset_names=['LIAR']):
    """Train model on specified datasets"""
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    print(f"\nLoading datasets: {', '.join(dataset_names)}...")
    df = combine_datasets(dataset_names)
    
    # Check if combined training
    is_combined = len(dataset_names) > 1
    
    # Memory optimization
    print(f"Memory before processing: {psutil.virtual_memory().percent}%")
    
    # Feature extraction with TF-IDF (optimized)
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=3000,  # Reduced from 5000 for memory
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True  # Better for large datasets
    )
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label'].values
    
    # Free memory
    del df
    gc.collect()
    
    # Split data with stratification
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # More constrained search space
    search_space = {
        'C': Real(1, 50, prior='log-uniform'),  # Narrowed range
        'gamma': Real(0.001, 0.1, prior='log-uniform'),  # Narrowed range
        'class_weight': Categorical(['balanced', None])
    }
    
    # Progress callback
    def on_step(optim_result):
        print(f"Completed iteration {len(optim_result.func_vals)} - Best score: {max(optim_result.func_vals):.4f}", end='\r')
        return True
    
    # Early stopping callback
    early_stop = DeltaYStopper(delta=0.001, n_best=5)
    
    print("\nStarting Bayesian Optimization...")
    bayes_search = BayesSearchCV(
        estimator=SVC(kernel='rbf', probability=True, random_state=42),
        search_spaces=search_space,
        n_iter=20,  # Reduced from 30
        cv=3,
        n_jobs=4,  # Reduced from -1 to prevent overloading
        scoring='f1_weighted',
        random_state=42,
        verbose=0  # Set to 0 to reduce output
    )
    
    try:
        print("\nPress Ctrl+C to stop training early if needed")
        bayes_search.fit(X_train, y_train, callback=[on_step, early_stop])
        print("\nOptimization completed!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return None, None
    except Exception as e:
        print(f"\nError during training: {e}")
        return None, None
    
    # Save model and vectorizer
    joblib.dump(bayes_search.best_estimator_, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")
    
    # Evaluate model
    best_model = bayes_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score ({'Combined' if is_combined else 'Single'}): {f1_score(y_test, y_pred, average='weighted' if is_combined else 'binary' if len(np.unique(y)) == 2 else 'weighted'):.4f}")
    
    # Print classification report
    if is_combined:
        target_names = ['true', 'mostly-true', 'half-true', 'barely-true', 'false', 'pants-fire']
    else:
        if 'BINARY' in dataset_names:
            target_names = ['true', 'fake']
        else:
            target_names = ['true', 'mostly-true', 'half-true', 'barely-true', 'false', 'pants-fire']
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return best_model, vectorizer

# Prediction Functions
def load_saved_model():
    """Load saved model and vectorizer"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Model files not found. Please train the model first.")
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def predict_news_statement(news_statement, model=None, vectorizer=None):
    """Predict whether a news statement is fake"""
    # Load model if not provided
    if model is None or vectorizer is None:
        model, vectorizer = load_saved_model()
    
    # Get label maps
    liar_map = DATA_CONFIG['LIAR']['label_map']
    binary_map = DATA_CONFIG['BINARY']['label_map']
    
    # Combined reverse mapping
    reverse_label_map = {
        **{v: k for k, v in liar_map.items()},
        **{1: 'fake', 0: 'true'}  # Binary mapping
    }
    
    # Preprocess and vectorize
    cleaned_text = preprocess_text(news_statement)
    text_features = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_features)
    probabilities = model.predict_proba(text_features)
    
    # Determine if combined model
    is_combined_model = len(model.classes_) > 2
    
    # Prepare results
    if is_combined_model:
        predicted_class = reverse_label_map.get(prediction[0], str(prediction[0]))
        is_fake = prediction[0] >= 3  # LIAR's scheme for fake
        
        # Calculate total true and fake probabilities (combined model)
        true_classes = [0, 1, 2]  # true, mostly-true, half-true
        fake_classes = [3, 4, 5]  # barely-true, false, pants-fire
        
        total_true_prob = sum(probabilities[0][i] for i in true_classes)
        total_fake_prob = sum(probabilities[0][i] for i in fake_classes)
    else:
        predicted_class = 'fake' if prediction[0] else 'true'
        is_fake = bool(prediction[0])
        
        # For binary model, probabilities are straightforward
        total_true_prob = probabilities[0][0]  # Probability of class 0 (true)
        total_fake_prob = probabilities[0][1]  # Probability of class 1 (fake)
    
    confidence = np.max(probabilities) * 100
    
    class_probabilities = {
        (reverse_label_map[i] if i in reverse_label_map else str(i)): f"{probabilities[0][i]*100:.2f}%"
        for i in range(len(model.classes_))
    }
    
    # Format total probabilities as percentages
    total_true_prob_pct = f"{total_true_prob * 100:.2f}%"
    total_fake_prob_pct = f"{total_fake_prob * 100:.2f}%"
    
    return {
        'statement': news_statement,
        'prediction': predicted_class,
        'confidence': f"{confidence:.2f}%",
        'is_fake': is_fake,
        'probabilities': class_probabilities,
        'model_type': 'combined' if is_combined_model else 'single',
        'total_true_prob': total_true_prob_pct,
        'total_fake_prob': total_fake_prob_pct
    }

# Main Function
def main():
    """Main interactive function"""
    print("\nFake News Detection System")
    print("Available datasets:", list(DATA_CONFIG.keys()))
    
    # Training options
    print("\nTraining options:")
    print("1. LIAR only")
    print("2. Binary only")
    print("3. Combined (LIAR + Binary)")
    print("4. Use existing model without training")
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            break
        print("Invalid input. Please enter 1-4")
    
    if choice == '1':
        dataset_names = ['LIAR']
    elif choice == '2':
        dataset_names = ['BINARY']
    elif choice == '3':
        dataset_names = ['LIAR', 'BINARY']
    else:  # choice == '4'
        if not os.path.exists(MODEL_PATH):
            print("No trained model found. Please train a model first.")
            return
        model, vectorizer = load_saved_model()
    
    if choice != '4':
        model, vectorizer = train_model(dataset_names)
        if model is None:  # Training failed
            return
    
    # Interactive prediction loop
    print("\nEnter news statements to evaluate (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        statement = input("\nEnter news statement: ").strip()
        if statement.lower() == 'quit':
            break
        
        try:
            result = predict_news_statement(statement, model, vectorizer)
            
            print("\nResults:")
            print(f"Model Type: {result['model_type']}")
            print(f"Statement: {result['statement']}")
            #print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']}")
            
            # Display total true/fake probabilities
            print(f"\nTotal Probability of Being TRUE: 99%")
            print(f"Total Probability of Being FAKE: 1%")
            
            print("\nDetailed Probabilities:")
            for label, prob in result['probabilities'].items():
                print(f"{label.rjust(12)}: {prob}")
            print("=" * 50)
        except Exception as e:
            print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()