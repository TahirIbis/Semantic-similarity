import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load('en_core_web_md')
nlp.add_pipe('spacytextblob')

amazon_df = pd.read_csv(r'amazon_product_reviews.csv')

index1 = int(input("Please enter a number between 0 and 41422: "))
index2 = int(input("Please enter a number between 0 and 41422: "))

if index1 > 41422 or index1 < 0:
    print("The 1st number you have entered is invalid") 
    exit()
if index2 > 41422 or index2 < 0:
    print("The 2nd number you have entered is not in the range")
    exit()

def preprocess_text(text):
    '''Preprocess text by lemmatizing and removing stopwords.'''
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def sentiment_analysis(text):
    doc = nlp(text)
    polarity = doc._.blob.polarity
    if polarity > 0:
        print("The reviews on this product are positive with a polarity score of: ", polarity)
    elif polarity == 0:
        print("The reviews on this product are neutral with a polarity score of: ", polarity)
    else:
        print("The reviews on this product are negative with a polarity score of: ", polarity)

# Selecting relevant columns and removing missing values
amazon_df = amazon_df['reviews.text'].dropna()
sample_1 = amazon_df[index1]
sample_2 = amazon_df[index2]
cleaned = preprocess_text(sample_1)
cleaned2 = preprocess_text(sample_2)
a = sentiment_analysis(cleaned)
b = sentiment_analysis(cleaned2)

similarity = nlp(cleaned).similarity(nlp(cleaned2))
print("Similarity Score between the 2 reviews are: ", similarity)
print()  # Empty line for readability