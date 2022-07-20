from constants import *
import pickle,pandas as pd,numpy as np
from numpy import *
import nltk,re,string
from textblob import TextBlob
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('tagsets')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')



class SentimentRecommendationModel:

    def __init__(self):

        try:
            self.model = pickle.load(open(ROOT_PATH + MODEL,'rb'))  
            self.vectorizer = pickle.load(open(ROOT_PATH +  VECTORIZER,'rb'))
            self.recommender = pickle.load(open(ROOT_PATH +  RECOMMENDER,'rb'))
            self.cleanDataset = pickle.load(open(ROOT_PATH +  DATASET,'rb'))
            self.data = pd.read_csv('Dataset/dataset.csv')
            self.lemmatizer = WordNetLemmatizer()
            self.stopWords = set(stopwords.words('english'))

        except IOError:
            print("Error: can\'t find file or read data")


    """create text processing function  """
    
    def textProcessing(text):
        '''
            This function parses a text and do the following.
            - Make the text lowercase
            - Remove whitespaces from both end of the string
            - Remove text in square brackets
            - Remove punctuation
            - Remove words containing numbers
        '''
        text = text.lower() # convert text to lower case
        text = text.strip() # remove whitespaces from both end of the string
        text = re.sub('\[.*?\]','',text) # remove text in square brackets
        text = re.sub('[%s]'%re.escape(string.punctuation),'',text) # remove string.punctuations(!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~) from the text
        #for removing punctuation you can also use str,maketrans(''.'',string.punctuation). it's just that re.sub is faster than maketrans
        text = re.sub('\w*\d\w*','',text) #remove words containing numbers
        return text

    """function to remove stop words from the text"""
    
    
    def removeStopwords(text,self):
        words = [word for word in text.split() if word.isalpha() and word not in self.stopWords]
        return " ".join(words)
        
    """helper function to map NTLK position tags"""
    
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    """remove stopwords and lematize text"""
    
    def textLemmatize(self,text):
        #remove the stopwords from the text
        words = self.removeStopwords(text)
        #map pos tags of each words
        wordnetPOSTags =  nltk.pos_tag(word_tokenize(words))
        #lemmatize the words according to their POS tag
        lemmatizedWords = [self.lemmatizer.lemmatize(token[0],self.get_wordnet_pos(token[1])) for i,token in enumerate(wordnetPOSTags)]
        return " ".join(lemmatizedWords)

    """function to extract the POS tags"""

    def pos_Tag(text):
    #TextBlob provides a simple API for diving into common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more
        blob = TextBlob(text)
        return " ".join([word for (word,tag) in blob.tags if tag in ['JJ','JJR','JJS','NN']])

    
    """function to classify the sentiment using the trained ML model"""

    def classify_sentiment(self, review_text):
        review_text = self.preprocess_text(review_text)
        X = self.vectorizer.transform([review_text])
        y_pred = self.model.predict(X)
        return y_pred

    """function to get the top product 20 recommendations for the user"""

    def getRecommendationByUser(self, user):
        recommendations = self.recommender.loc[user].sort_values(ascending=False)[0:20]
        recommendations = pd.DataFrame({'id': recommendations.index, 'similarity_score' : recommendations})
        return recommendations.reset_index(drop=True)

    """function to filter the product recommendations using the sentiment model and get the top 5 recommendations"""

    def getTop5PRoductsToRecommend(self,user):
        if (user in self.user_final_rating.index):
            recommendations = self.getRecommendationByUser(user)
            filtered_data =  self.cleanDataset(self.cleanDataset.id.isin(recommendations.id))
            X= self.vectorizer.transform(filtered_data.finalReviews.values.astype(str))
            filtered_data['predicted_sentiment'] = self.model.predict(X)
            df = pd.merge(recommendations, filtered_data, on="id").drop_duplicates()
            temp = df.drop(columns=['id','reviews_username','reviewsText','finalReviews','similarity_score'])
            df_grouped  = temp.groupby('name', as_index=False).count()
            df_grouped['similarity_score'] = df_grouped.name.apply(lambda x: df[(df.name==x) & (df['predicted_sentiment']==1)]["similarity_score"].median())
            df_grouped["pos_review_count"] = df_grouped.name.apply(lambda x: df[(df.name==x) & (df['predicted_sentiment']==1)]["predicted_sentiment"].count())
            df_grouped["total_review_count"] = df_grouped['predicted_sentiment']
            df_grouped['pos_sentiment_percent'] = np.round(df_grouped["pos_review_count"]/df_grouped["total_review_count"]*100,2)
            return df_grouped.sort_values(['pos_sentiment_percent','name'], ascending=[False,True])[0:5]
        else:
            print(f"User name {user} doesn't exist")
            return None

    
if __name__== '__main__':
    print("arpit")
    classObj = SentimentRecommendationModel()

