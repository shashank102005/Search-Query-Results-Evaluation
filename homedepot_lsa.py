# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:10:16 2016

@author: Shashank
"""

import pandas as pd
import numpy as np
import nltk
from nltk import PorterStemmer
from nltk import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics 
from sklearn.neighbors import DistanceMetric
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
import math
import gensim
from gensim import corpora, models, similarities
from scipy.linalg import norm
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from scipy.stats import entropy
from sklearn.decomposition import NMF   
   
   
lmtzr = WordNetLemmatizer()   


########## function for pre processing of strings
def process_strings( string ):
    # 1. Remove HTML
    words = BeautifulSoup(string).get_text()
    
       
    # separate joint words
    words = re.sub('(\w+)([A-Z][a-z]+)',lambda m: " " + m.group(1) +\
               " " + m.group(2),  words  )
    
    # 3. Convert to lower case
    words = words.lower() 
    
    # remove unwanted characters
    ddd = re.sub('[^a-zA-Z\s]', " ", words )
    ddd2 = re.sub( "(\d+)x(\d+)", lambda m: m.group(1) + " " + m.group(2)  , ddd )
    ddd3 = re.sub( "(\d+)x\s", lambda m: m.group(1) + " ", ddd2 )
    ddd4 = re.sub( "\sx(\d+)", lambda m:  " " + m.group(1), ddd3 )
    ddd5 = re.sub( "\sx\s",  " " , ddd4 )
    fff = re.sub( "(\D+)(\d+)", lambda m:  m.group(1) + " " + m.group(2), ddd5 ) 
    fff2 = re.sub( "(\d+)(\D+)", lambda m:  m.group(1) + " " + m.group(2), fff )
    words = re.sub( "(\d+)(\D+)(\d+)", lambda m:  m.group(1) + " " + m.group(2) + " " \
                  + m.group(3), fff2)
    for  i in range(1,10):   
      words = re.sub('\s(ft|sq|in|gal|cu|h|oz|dia|yd|yds|a|p|qt|ah|amp|gpm|mp\
                       |quart|watt|cc|d|inc|incl|lb|lbs|lin|ln|mil|mm|no|n|oc\
                       |od|pc|pal|pt|s|sch|cs|case|pallet|w)\s'  , lambda m: " ", words )
        
    words = words.split()                             
   
    stops = set(stopwords.words("english"))                  
   
    meaningful_words = [w for w in words if not w in stops] 
    
    lemm_words = [lmtzr.lemmatize(w) for w in meaningful_words]
    
    return(" ".join(lemm_words))
        
       



##################### read the train data
train_data =pd.read_csv('/media/shashank/Data/Projects/Kaggle/Home Depot/train.csv',encoding = "ISO-8859-1")

##################### read the product descriptions data
proddesc_data =pd.read_csv('/media/shashank/Data/Projects/Kaggle/Home Depot/product_descriptions.csv',\
                        encoding = "ISO-8859-1")
                      
##################### read the attribute  data
attr_data =pd.read_csv('/media/shashank/Data/Projects/Kaggle/Home Depot/attributes.csv',\
                        encoding = "ISO-8859-1")

##################### read the attribute data frame
attr_dfcsv =pd.read_csv('/media/shashank/Data/Projects/Kaggle/Home Depot/data/attr_df.csv',\
                        encoding = "ISO-8859-1")
attr_dfcsv =attr_dfcsv.drop(['Unnamed: 0'],axis=1)


################### merge the train and attribue
train_attr = pd.merge(left=train_data,right=attr_dfcsv,how='left',left_on='product_uid',\
                          right_on='product_uid')
                          

################# modify attr_dfcsv
attr_df = attr_dfcsv
attr_df = attr_df.set_index('product_uid')
sno = range(0,attr_df.shape[0])
attr_df['sno'] = sno
                          


###################### clean the product titles and search queries 
clean_alldata =[]
 
for i in range( 0, len(train_data) ):
     clean_alldata.append( process_strings(str(train_data["product_title"][i]) ) )

for i in range( 0, len(train_data) ):
     clean_alldata.append( process_strings(str(train_data["search_term"][i]) ) )

for i in range( 0, len(proddesc_data)):
     clean_alldata.append( process_strings( str(proddesc_data["product_description"][i]) ) ) 
     
for i in range( 0, len(attr_df)):
     clean_alldata.append( process_strings(str(attr_dfcsv["value"][i])))    





####### create tf-idf vectors for cleaned  product titles , desc and search queries  
vectorizer1 = TfidfVectorizer(analyzer = "word", binary = False, norm = 'l2', )
tfidf_features = vectorizer1.fit_transform(clean_alldata)


 
########### Latent Semantic Analysis  Allocation
lsa = TruncatedSVD(n_components=100,algorithm='arpack').fit_transform(tfidf_features)



############ Isolate the LSA components for query,title,description and attributes
lsa_title = lsa[0:len(train_data)]
lsa_search_term = lsa[len(train_data):2*len(train_data)]
lsa_prodc_desc = lsa[2*len(train_data):2*len(train_data)+len(proddesc_data)]
lsa_attr = lsa[2*len(train_data)+len(proddesc_data):2*len(train_data)\
                                            +len(proddesc_data)+len(attr_dfcsv)]



########### build data frame of LSA component scores
lsa_search_term_df = pd.DataFrame(lsa_search_term,columns=np.arange(0, 100))
lsa_title_df = pd.DataFrame(lsa_title,columns=np.arange(100, 200))

lsa_prodc_desc2 = np.zeros(shape=lsa_title.shape)
for i in range(0,len(lsa_title)):
    lsa_prodc_desc2[i] = lsa_prodc_desc[train_data['product_uid'][i]-100001]
lsa_prodc_desc2_df = pd.DataFrame(lsa_prodc_desc2,columns=np.arange(200, 300))    
    
    

lsa_attr2 = np.zeros(shape=lsa_title.shape)
for i in range(0,len(lsa_title)):
    if (pd.isnull(train_attr['value'][i])):
      lsa_attr2[i] = -999
    else:
      puid = train_attr['product_uid'][i]
      ind = attr_df['sno'][puid]
      lsa_attr2[i] = lsa_attr[ind]
lsa_attr2_df = pd.DataFrame(lsa_attr2,columns=np.arange(300, 400))
      
       
lsa_df = pd.concat([lsa_search_term_df,lsa_title_df,lsa_prodc_desc2_df,lsa_attr2_df],axis=1)
lsa_df.columns



################## compute document similarity using LSA components
lsa_sim_title = np.zeros(shape=len(lsa_df))
for i in range(0,len(lsa_df)):
  lsa_sim_title[i] = np.asmatrix(lsa_search_term[i]) * np.asmatrix(lsa_title[i]).T   

lsa_sim_desc = np.zeros(shape=len(lsa_df))
for i in range(0,len(lsa_df)):
  lsa_sim_desc[i] = np.asmatrix(lsa_search_term[i]) * np.asmatrix(lsa_prodc_desc2[i]).T   

lsa_sim_attr = np.zeros(shape=len(lsa_df))
for i in range(0,len(lsa_df)):
  if(lsa_attr2[i,0] == -999 ):  
    lsa_sim_attr[i] = -999
  else:
    lsa_sim_attr[i] = np.asmatrix(lsa_search_term[i]) * np.asmatrix(lsa_attr2[i]).T 



######## cosine similarity 
lsa_cosine_sim_title = np.zeros(shape=len(lsa_df))
for i in range(0,len(lsa_df)):
   lsa_cosine_sim_title[i] = cosine_similarity(lsa_search_term[i],lsa_title[i])

lsa_cosine_sim_desc = np.zeros(shape=len(lsa_df))
for i in range(0,len(lsa_df)):
   lsa_cosine_sim_desc[i] = cosine_similarity(lsa_search_term[i],lsa_prodc_desc2[i])

lsa_cosine_sim_attr = np.zeros(shape=len(lsa_df))
for i in range(0,len(lsa_df)):
   if(lsa_attr2[i,0] == -999 ):  
     lsa_cosine_sim_attr[i] = -999
   else:
     lsa_cosine_sim_attr[i] = cosine_similarity(lsa_search_term[i],lsa_attr2[i])    





###################### construct a data frame for LSA features
lsa_features_df = pd.DataFrame({'lsa_sim_title':lsa_sim_title,
                                'lsa_sim_desc':lsa_sim_desc,
                                'lsa_sim_attr':lsa_sim_attr,
                                'lsa_cosine_sim_title':lsa_cosine_sim_title,
                                'lsa_cosine_sim_desc':lsa_cosine_sim_desc,
                                'lsa_cosine_sim_attr':lsa_cosine_sim_attr
                                })






lsa_df.to_csv('lsa_df.csv')
lsa_features_df.to_csv('lsa_features_df.csv')







