# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:49:35 2016

@author: shashank
"""

import pandas as pd
import numpy as np
import nltk
import nltk.data
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
import math
import xgboost as xgb
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging
from gensim.models import word2vec
from pyemd import emd


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



############### function for pre processing of strings ##################################
def process_strings( string ):
    # 1. Remove HTML
    words = BeautifulSoup(string).get_text()
    
       
    # separate joint words
    words = re.sub('(\w+)([A-Z][a-z]+)',lambda m:  " " + m.group(1) +\
               " " + m.group(2),  words  )
    
    # 3. Convert to lower case
    words = words.lower() 
    
    # remove unwanted characters
    ddd = re.sub('[^a-zA-Z0-9\s]', " ", words )
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
        
    # Join the words back into one string separated by space
    return (  words.split() )
    
##########################################################################################
    

#######################  Spliting long paragraph into sentences ###########################

def doc_to_sentences(doc):
    raw_sentences = tokenizer.tokenize(doc.strip())
   
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
              sentences.append( process_strings( raw_sentence))
    
    return sentences
#######################################################################################



##################### read the train data
train_data =pd.read_csv('/media/shashank/Data/Projects/Kaggle/Home Depot/train.csv',\
                        encoding = "ISO-8859-1")

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


#################### merge the product description and train data ( left outer join)
train_proddesc = pd.merge(left=proddesc_data,right=train_data,left_on='product_uid',\
                          right_on='product_uid')



################### merge the train and attribue
train_attr = pd.merge(left=train_data,right=attr_dfcsv,how='left',left_on='product_uid',\
                          right_on='product_uid')    


################# modify attr_dfcsv
attr_df = attr_dfcsv
attr_df = attr_df.set_index('product_uid')
sno = range(0,attr_df.shape[0])
attr_df['sno'] = sno



###################### clean and tokenize the product titles, descriptions  and search queries 
clean_alldata =[]
 
for i in range( 0, len(train_data) ):
     clean_alldata.append( process_strings(str(train_data["product_title"][i])))

for i in range( 0, len(train_data) ):
     clean_alldata.append( process_strings(str(train_data["search_term"][i])))

for i in range( 0, len(proddesc_data)):
     clean_alldata += doc_to_sentences( str(proddesc_data["product_description"][i])) 
         
for i in range( 0, len(attr_df)):
     clean_alldata += doc_to_sentences( str(process_strings(attr_dfcsv["value"][i])))
##########################################################################################



########################## word2vec model of the extracted word corpus ###################

# Set values for various parameters
num_features = 500    # Word vector dimensionality                      
min_word_count = 0  # Minimum word count                        
num_workers = 2       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)

print ("Training model...")
model = word2vec.Word2Vec(clean_alldata, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)


model.init_sims(replace=True)


model_name = "500features_0minwords_10context"
model.save(model_name)

##############################################################################################



###################### compute cosine similarities between the search term and other items ###
word2vec_cosine_sim_title = np.zeros(shape=len(train_data))
for i in range(0,len(train_data)):
  word2vec_cosine_sim_title[i] = model.n_similarity(process_strings(train_data['search_term'][i]),\
                    process_strings(train_data['product_title'][i]))


word2vec_cosine_sim_desc = np.zeros(shape=len(train_data))
for i in range(0,len(train_data)):
  word2vec_cosine_sim_desc[i] = model.n_similarity(process_strings(train_data['search_term'][i]),\
                    process_strings(proddesc_data['product_description'][train_data['product_uid'][i]-100001]))

 
word2vec_cosine_sim_attr = np.zeros(shape=len(train_data))
for i in range(0,len(train_data)):
   if (pd.isnull(train_attr['value'][i])):
      word2vec_cosine_sim_attr[i] = -999
   else:
      puid = train_attr['product_uid'][i]
      ind = attr_df['sno'][puid]
      word2vec_cosine_sim_attr[i] = model.n_similarity(process_strings(train_data['search_term'][i]),\
                    process_strings(attr_dfcsv['value'][ind]))


############################################################################################
 


####################### compute word mover's distances ######################################

wmd_title = np.zeros(shape=len(train_data))

for i in range(0,len(train_data)): 
  search = process_strings(train_data['search_term'][i])
  title = process_strings(train_data['product_title'][i])

  min_arr = np.zeros(shape = len(search))
  dis_arr = np.zeros(shape = len(title))

  for k in range(0,len(search)):
      for j in range(0,len(title)):
         dis_arr[j] = 1 - model.n_similarity(search[k],title[j])   
      min_arr[k] = dis_arr.min()
        
  wmd_title[i]=min_arr.sum()



wmd_desc = np.zeros(shape=len(train_data))

for i in range(0,len(train_data)): 
  search = process_strings(train_data['search_term'][i])
  desc = process_strings(proddesc_data['product_description'][train_data['product_uid'][i]-100001])

  min_arr = np.zeros(shape = len(search))
  dis_arr = np.zeros(shape = len(desc))

  for k in range(0,len(search)):
      for j in range(0,len(desc)):
         dis_arr[j] = 1 - model.n_similarity(search[k],desc[j])   
      min_arr[k] = dis_arr.min()
        
  wmd_desc[i]=min_arr.sum()



wmd_attr = np.zeros(shape=len(train_data))

for i in range(0,len(train_data)): 
    
   if (pd.isnull(train_attr['value'][i])):  
       wmd_attr[i]=-999
   else:    
     search = process_strings(train_data['search_term'][i])
  
     puid = train_attr['product_uid'][i]
     ind = attr_df['sno'][puid]
     attr = process_strings(attr_dfcsv['value'][ind])
  
     min_arr = np.zeros(shape = len(search))
     dis_arr = np.zeros(shape = len(attr))

     for k in range(0,len(search)):
       for j in range(0,len(attr)):
          dis_arr[j] = 1 - model.n_similarity(search[k],attr[j])   
       min_arr[k] = dis_arr.min()
          
     wmd_attr[i]=min_arr.sum()
    
   
   
    

#########################################################################################





















