import pandas as pd
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from rank_bm25 import BM25Okapi
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Create an unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define stopwords and add custom stopwords
stop_words = set(stopwords.words('english') + list(string.punctuation))

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(filtered_tokens)

# Load the data
url = 'https://github.com/FeiYee/HerbKG/raw/main/DiseaseData/test.tsv'
data = pd.read_csv(url, sep='\t', header=None, names=['Text'], quoting=3, encoding='utf-8')

# Function to extract relevant information
def extract_info(text):
    disease_pattern = re.compile(r'Disease:\s*(.*?)\s*Plant:')
    plant_pattern = re.compile(r'Plant:\s*(.*?)\s*Usage:')
    usage_pattern = re.compile(r'Usage:\s*(.*)')

    disease = disease_pattern.search(text)
    plant = plant_pattern.search(text)
    usage = usage_pattern.search(text)

    if disease and plant:
        if usage:
            return disease.group(1), plant.group(1), usage.group(1)
        else:
            return disease.group(1), plant.group(1), ''
    return None, None, None

# Apply extraction to the dataset
extracted_data = data['Text'].apply(lambda x: pd.Series(extract_info(x), index=['Disease', 'Plant_Name', 'Usage']))
df = extracted_data.dropna().reset_index(drop=True)

# Check if data is extracted correctly
print(df.head())

# Preprocess the text
df['Disease'] = df['Disease'].apply(preprocess_text)
df['Plant_Name'] = df['Plant_Name'].apply(preprocess_text)
df['Usage'] = df['Usage'].apply(preprocess_text)

# Check if preprocessing is done correctly
print(df.head())

# Create a corpus of diseases
corpus = df['Disease'].tolist()

# Tokenize the corpus
tokenized_corpus = [doc.split() for doc in corpus]

# Create BM25 model
if tokenized_corpus:  # Ensure corpus is not empty
    bm25 = BM25Okapi(tokenized_corpus)

    # Function to get plant name and usage for a given disease
    def get_plant_info(disease):
        tokenized_disease = preprocess_text(disease).split()
        scores = bm25.get_scores(tokenized_disease)
        best_doc_idx = scores.argmax()
        return df.iloc[best_doc_idx]['Plant_Name'], df.iloc[best_doc_idx]['Usage']

    # Example test data for evaluation
    test_data = [
        {"disease": "Alzheimer's disease", "expected_plant": "gx-50 derived from Zanthoxylum bungeanum", "expected_usage": ""},
        # Add more test cases
    ]

    # Initialize lists to store results
    y_true_plant = []
    y_pred_plant = []
    y_true_usage = []
    y_pred_usage = []

    # Test the model and collect results
    for test in test_data:
        disease = preprocess_text(test['disease'])
        expected_plant = preprocess_text(test['expected_plant'])
        expected_usage = preprocess_text(test['expected_usage'])
        plant_name, usage = get_plant_info(disease)
        
        y_true_plant.append(expected_plant)
        y_pred_plant.append(plant_name)
        y_true_usage.append(expected_usage)
        y_pred_usage.append(usage)

    # Calculate metrics for plant name prediction
    precision_plant = precision_score(y_true_plant, y_pred_plant, average='weighted', zero_division=1)
    recall_plant = recall_score(y_true_plant, y_pred_plant, average='weighted', zero_division=1)
    f1_plant = f1_score(y_true_plant, y_pred_plant, average='weighted', zero_division=1)
    accuracy_plant = accuracy_score(y_true_plant, y_pred_plant)

    # Calculate metrics for usage prediction (handle empty expected_usage)
    if any(y_true_usage):
        precision_usage = precision_score(y_true_usage, y_pred_usage, average='weighted', zero_division=1)
        recall_usage = recall_score(y_true_usage, y_pred_usage, average='weighted', zero_division=1)
        f1_usage = f1_score(y_true_usage, y_pred_usage, average='weighted', zero_division=1)
        accuracy_usage = accuracy_score(y_true_usage, y_pred_usage)
    else:
        precision_usage = 0.0
        recall_usage = 0.0
        f1_usage = 0.0
        accuracy_usage = 0.0

    # Print results
    print(f"Plant Name Prediction - Precision: {precision_plant}, Recall: {recall_plant}, F1 Score: {f1_plant}, Accuracy: {accuracy_plant}")
    print(f"Usage Prediction - Precision: {precision_usage}, Recall: {recall_usage}, F1 Score: {f1_usage}, Accuracy: {accuracy_usage}")
else:
    print("The corpus is empty, please check the data extraction and preprocessing steps.")
