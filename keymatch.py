#USING GIT REPO- HERBKG

import pandas as pd 
#intalling natural language pack
import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#preprocessing- cleans and standardises the data
def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words]
    words = [word for word in words if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

#info retrieval
def retrieve_information(document, topic):
    processed_document = preprocess_text(document)
    processed_topic = preprocess_text(topic)
    sentences = sent_tokenize(document)
    relevant_sentences = []
    for sentence in sentences:
        processed_sentence = preprocess_text(sentence)
        if any(word in processed_sentence for word in processed_topic):
            relevant_sentences.append(sentence)
    
    return relevant_sentences

url='https://github.com/FeiYee/HerbKG/raw/main/DiseaseData/test.tsv'
#RAW LINK FROM THE GIT PROVIDED BY MA'AM

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