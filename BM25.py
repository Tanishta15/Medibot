import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function - cleans and standardizes the data
def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words]
    words = [word for word in words if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Information retrieval function using BM25
def retrieve_information(document, topic):
    sentences = sent_tokenize(document)
    tokenized_sentences = [preprocess_text(sentence) for sentence in sentences]
    
    bm25 = BM25Okapi(tokenized_sentences)
    
    processed_topic = preprocess_text(topic)
    relevant_scores = bm25.get_scores(processed_topic)
    
    threshold = 0.1  # This threshold can be adjusted based on requirement
    relevant_sentences = [sentences[i] for i in range(len(sentences)) if relevant_scores[i] > threshold]
    
    return relevant_sentences

# Fetch the document from the URL using pandas
url = "https://github.com/FeiYee/HerbKG/raw/main/DiseaseData/test.tsv"
df = pd.read_csv(url, sep='\t')
document = ' '.join(df.astype(str).apply(lambda x: ' '.join(x), axis=1))

topic = input("Enter the topic: ")

relevant_info = retrieve_information(document, topic)

if relevant_info:
    print("Relevant information:")
    for sentence in relevant_info:
        print(sentence)
else:
    print("No relevant information found on the topic.")