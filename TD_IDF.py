import pandas as pd 
import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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
    return ' '.join(words)

# Information retrieval function using TF-IDF
def retrieve_information(document, topic):
    sentences = sent_tokenize(document)
    processed_sentences = [preprocess_text(sentence) for sentence in sentences]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)
    
    processed_topic = preprocess_text(topic)
    topic_vector = vectorizer.transform([processed_topic])
    
    cosine_similarities = np.dot(tfidf_matrix, topic_vector.T).toarray()
    
    relevant_sentences = [sentences[i] for i in range(len(sentences)) if cosine_similarities[i] > 0.1]
    
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