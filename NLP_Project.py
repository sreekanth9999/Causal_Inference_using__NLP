##code starts with importing all nlp modules required for execution at later levels

from os import listdir
from os.path import isfile, join
import pandas as pd
import os
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from html.parser import HTMLParser
import xml
import nltk
import html.parser    
from bs4 import BeautifulSoup
from gensim.models.phrases import Phrases , Phraser
import pickle


#reading all documents in the folder
mypath='E:/CAUSAL_PROJECT/DATA/DUMMY/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
doc_complete = [ ' '.join(open(join(mypath, f), 'r').read().split()[1:]) for f in onlyfiles]
#extracting ciks from doc names
cik_codes=pd.DataFrame(onlyfiles,columns={'cik'})
cik_codes['cik'].str.split('.')
cik_codes_1 = pd.DataFrame(cik_codes.cik.str.split('.',1).tolist(),columns = ['cik','type'])
##removing all html tags from all txt documents
doc_with_cleanhtml=[html.parser.HTMLParser().unescape(doc) for doc in doc_complete]

#removing all html tags from all txt documents
cleantext = [BeautifulSoup(doc).text for doc in doc_with_cleanhtml]

##removing all single letters from documents
import re 
text_removecharacters=[re.sub(r'[^\w]', ' ', doc) for doc in cleantext]

## removing words if not an english words ie, MFgwCgYEVQgBAQICAf8DSgAwRwJAW2sNKK9AVtBzYZmr6aGjlWyK3XmZv3dTINen  such kind of words
words = set(nltk.corpus.words.words())

def document(text):
    print (len(text))
    data= " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())
    print ("AFTERRRR",len(data))
    
    return data
    
text_with_english_letters=[document(doc) for doc in text_removecharacters]

##removing stop words,alphanumeric characters, stemming and removing all numbers from docs

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
stemmer = SnowballStemmer("english")
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(stemmer.stem(word) for word in punc_free.split())
    result = ''.join([i for i in normalized if not i.isdigit()])
    #
                      
    
    
    
    return result

doc_clean = [clean(doc).split() for doc in text_with_english_letters]
stemmed_text=[" ".join(doc) for doc in doc_clean]
data123=stemmed_text.copy()

#removing all two letter words from docs ..can remove 3 letters but words like tax gets removed
final_data_after=[re.sub(r'\b\w{1,2}\b', '', data) for data in data123]
doc_clean_1 = [clean(doc).split() for doc in final_data_after]

phrases = Phrases(doc_clean_1)
bigram = Phrases(doc_clean_1, min_count=1, delimiter=b' ')
trigram = Phrases(bigram[doc_clean_1], min_count=1, delimiter=b' ')

def document123(doc):
    bigrams_ = [b for b in bigram[doc] if b.count(' ') == 1]
    trigrams_ = [t.replace(" ","_") for t in trigram[bigram[doc]] if t.count(' ') <= 2]

    #print(bigrams_)
    return trigrams_

final_data_demo12345=[document123(doc) for doc in doc_clean_1]
final_data12=[" ".join(doc) for doc in final_data_demo12345]

final_data_demo12345=[document123(doc) for doc in doc_clean_1]

final_data12=[" ".join(doc) for doc in final_data_demo12345]

def demo(data):
    
    return " ".join([i for i in data.lower().split() if i.find("_") != -1 ]) 
final_data=[demo(doc) for doc in final_data12]


#converting into a matrix using countvector mtd having docments on y axis and ngram words on x axis with frequencies
cv2 = CountVectorizer()
cv_fit2=cv2.fit_transform(final_data).toarray()
required_data2=pd.DataFrame(cv_fit2,columns=cv2.get_feature_names())

required_data2.insert(0, 'cik_dataaa12345', cik_codes_1['cik'])
required_data3=required_data2.drop(required_data2.columns[required_data2.apply(lambda col: (col==0).astype(int).sum(axis=0) > 0.98*len(required_data2))], axis=1)

required_data3['cik_dataaa12345']=required_data3['cik_dataaa12345'].astype(float)
required_data2.to_csv('E://CAUSAL_PROJECT/DATA/OUTPUTS/demo_4.csv')
dta_file=pd.io.stata.read_stata('E://CAUSAL_PROJECT/DATA/McClane_IPO_data_assembled.dta')
dta_file.to_csv("E://CAUSAL_PROJECT/DATA/OUTPUTS/stata_to_csv_1.csv")
dta_file=pd.read_csv("E://CAUSAL_PROJECT/DATA/OUTPUTS/stata_to_csv.csv")
dataframe_with_sic_year=dta_file[['cik','sic','sic_category','ipoyear','managercounsel','issuercounsel']]
dataframe_with_sic_year_1=dataframe_with_sic_year[dataframe_with_sic_year['cik'].notnull()].reset_index()
dataframe_with_sic_year_1=dataframe_with_sic_year_1.drop('index',axis=1)
dataframe_with_sic_year_1['cik']=dataframe_with_sic_year_1['cik'].astype(float)
dataframe_with_sic_year_1.rename(columns={'cik':'cik_dataaa12345'}, inplace = True)

##final dataframe after matrix conversion
data_requires_for_jeremy=pd.merge(dataframe_with_sic_year_1,required_data2,on="cik_dataaa12345",how='inner')
data_requires_for_jeremy.to_csv('E://CAUSAL_PROJECT/DATA/OUTPUTS/SIC_Year_Data.csv')

##elemiination columns which has 80% of the rows filled with zeros(****) can change as per needeed
df=data_requires_for_jeremy.copy()
Data_frame_final=df.drop(df.columns[df.apply(lambda col: (col==0).astype(int).sum(axis=0) > 0.80*len(df))], axis=1)


####part_1 ends##############################
