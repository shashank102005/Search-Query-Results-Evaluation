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
    
    return(  meaningful_words )
        
       



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
alldata =[]
 
for i in range( 0, len(train_data) ):
    alldata.append( str(train_data["product_title"][i]))

for i in range( 0, len(train_data) ):
     alldata.append( str(train_data["search_term"][i]) ) 

for i in range( 0, len(proddesc_data)):
     alldata.append(  str(proddesc_data["product_description"][i]) ) 
     
for i in range( 0, len(attr_dfcsv)):
     alldata.append( str(attr_dfcsv["value"][i]))   



####################### list for tokenized documents in loop
texts = []

en_stop = get_stop_words('en')
tokenizer = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()

for i in alldata:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop ]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)




################################ turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    

################################# convert tokenized documents into a document-term matrix
docbow = [dictionary.doc2bow(text) for text in texts]



####### create bag of words for cleaned  product titles , desc and search queries  
tfidf = models.TfidfModel(docbow)
corpus_tfidf = tfidf[docbow] 
 
 
########### Latent Dirichlet Allocation
lda = gensim.models.ldamodel.LdaModel(corpus=docbow , id2word=dictionary, num_topics=100,\
                    update_every=1, chunksize=10000, passes=10)

def list_to_matrix(lst):
    return [lst]

doc_topic = np.zeros(shape=(len(docbow),100))
for i in range(0,len(docbow)):
  temp = lda.get_document_topics(docbow[i], minimum_probability=0.00001)
  doc_topic[i,:] = np.asarray(temp)[:,1]



############ Isolate the lda components for query,title,description and attributes
lda_title = doc_topic[0:len(train_data)]
lda_search_term = doc_topic[len(train_data):2*len(train_data)]
lda_prodc_desc = doc_topic[2*len(train_data):2*len(train_data)+len(proddesc_data)]
lda_attr = doc_topic[2*len(train_data)+len(proddesc_data):2*len(train_data)\
                                            +len(proddesc_data)+len(attr_dfcsv)]



########### build data frame of LDA component scores
lda_search_term_df = pd.DataFrame(lda_search_term,columns=np.arange(0, 100))
lda_title_df = pd.DataFrame(lda_title,columns=np.arange(100, 200))

lda_prodc_desc2 = np.zeros(shape=lda_title.shape)
for i in range(0,len(lda_title)):
    lda_prodc_desc2[i] = lda_prodc_desc[train_data['product_uid'][i]-100001]
lda_prodc_desc2_df = pd.DataFrame(lda_prodc_desc2,columns=np.arange(200, 300))    
    
    

lda_attr2 = np.zeros(shape=lda_title.shape)
for i in range(0,len(lda_title)):
    if (pd.isnull(train_attr['value'][i])):
      lda_attr2[i] = -999
    else:
      puid = train_attr['product_uid'][i]
      ind = attr_df['sno'][puid]
      lda_attr2[i] = lda_attr[ind]
lda_attr2_df = pd.DataFrame(lda_attr2,columns=np.arange(300, 400))
      
       
lda_df = pd.concat([lda_search_term_df,lda_title_df,lda_prodc_desc2_df,lda_attr2_df],axis=1)
lda_df.columns



################## compute document similarity using lda components
lda_sim_title = np.zeros(shape=len(lda_df))
for i in range(0,len(lda_df)):
  lda_sim_title[i] = np.asmatrix(lda_search_term[i]) * np.asmatrix(lda_title[i]).T   

lda_sim_desc = np.zeros(shape=len(lda_df))
for i in range(0,len(lda_df)):
  lda_sim_desc[i] = np.asmatrix(lda_search_term[i]) * np.asmatrix(lda_prodc_desc2[i]).T   

lda_sim_attr = np.zeros(shape=len(lda_df))
for i in range(0,len(lda_df)):
  if(lda_attr2[i,0] == -999 ):  
    lda_sim_attr[i] = -999
  else:
    lda_sim_attr[i] = np.asmatrix(lda_search_term[i]) * np.asmatrix(lda_attr2[i]).T 



######## cosine similarity - matching
lda_cosine_sim_title = np.zeros(shape=len(lda_df))
for i in range(0,len(lda_df)):
   lda_cosine_sim_title[i] = cosine_similarity(lda_search_term[i],lda_title[i])

lda_cosine_sim_desc = np.zeros(shape=len(lda_df))
for i in range(0,len(lda_df)):
   lda_cosine_sim_desc[i] = cosine_similarity(lda_search_term[i],lda_prodc_desc2[i])

lda_cosine_sim_attr = np.zeros(shape=len(lda_df))
for i in range(0,len(lda_df)):
   if(lda_attr2[i,0] == -999 ):  
     lda_cosine_sim_attr[i] = -999
   else:
     lda_cosine_sim_attr[i] = cosine_similarity(lda_search_term[i],lda_attr2[i])    



###################### Bhattacharya Distance
lda_bhat_sim_title = np.zeros(shape=len(lda_df))
for i in range(0,len(lda_df)):
   lda_bhat_sim_title[i] = norm(np.sqrt(lda_search_term[i]) - np.sqrt(lda_title[i]))

lda_bhat_sim_desc = np.zeros(shape=len(lda_df))
for i in range(0,len(lda_df)):
   lda_bhat_sim_desc[i] = norm(np.sqrt(lda_search_term[i]) - np.sqrt(lda_prodc_desc2[i])) 

lda_bhat_sim_attr = np.zeros(shape=len(lda_df))
for i in range(0,len(lda_df)):
   if(lda_attr2[i,0] == -999 ):  
     lda_bhat_sim_attr[i] = -999
   else:
     lda_bhat_sim_attr[i] = norm(np.sqrt(lda_search_term[i]) - np.sqrt(lda_attr2[i]))  



##################### Kullback Liebler Divergence
lda_kld_title = np.zeros(shape=len(lda_df))
for i in range(0,len(lda_df)):
 lda_kld_title[i] = entropy(pk=lda_search_term[i], qk=lda_title[i])

lda_kld_desc = np.zeros(shape=len(lda_df))
for i in range(0,len(lda_df)):
 lda_kld_desc[i] = entropy(pk=lda_search_term[i], qk=lda_prodc_desc2[i])


lda_kld_attr = np.zeros(shape=len(lda_df))
for i in range(0,len(lda_df)):
   if(lda_attr2[i,0] == -999 ):  
     lda_kld_attr[i] = -999
   else:  
     lda_kld_attr[i] = entropy(pk=lda_search_term[i], qk=lda_attr[i])






###################### construct a data frame for lda features
lda_features_df = pd.DataFrame({'lda_sim_title':lda_sim_title,
                                'lda_sim_desc':lda_sim_desc,
                                'lda_sim_attr':lda_sim_attr,
                                'lda_cosine_sim_title':lda_cosine_sim_title,
                                'lda_cosine_sim_desc':lda_cosine_sim_desc,
                                'lda_cosine_sim_attr':lda_cosine_sim_attr,
                                'lda_bhat_sim_title':lda_bhat_sim_title,
                                'lda_bhat_sim_desc':lda_bhat_sim_desc,
                                'lda_bhat_sim_attr':lda_bhat_sim_attr,
                                'lda_kld_title':lda_kld_title,
                                'lda_kld_desc':lda_kld_desc,
                                'lda_kld_attr':lda_kld_attr,
                                })






lda_df.to_csv('lda_df.csv')
lda_features_df.to_csv('lda_features_df.csv')








