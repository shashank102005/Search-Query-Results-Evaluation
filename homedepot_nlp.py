# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 17:22:48 2016

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
import math
import xgboost as xgb
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

    
############### function for pre processing of strings ##################################
def process_strings( string ):
    # 1. Remove HTML
    words = BeautifulSoup(string).get_text()
    
       
    # separate joint words
    words = re.sub('(\w+)([A-Z][a-z]+)',lambda m: m.group(0) + " " + m.group(1) +\
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
    return (  words )
    
##########################################################################################


######################## functions and variables for hyper parameter optimization #######
def score(params):
    print ("Training with params : ")
    print (params)
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dvalid = xgb.DMatrix(X_val, label=Y_val)
    model = xgb.train(params, dtrain, num_round)
    pred= model.predict(dvalid)
    score = mean_squared_error(Y_val, pred)**0.5 
    print ("\tScore {0}\n\n".format(score))
    return {'loss': score, 'status': STATUS_OK}

space = {
             'n_estimators' : hp.quniform('n_estimators', 100, 1000, 1),
             'eta' : hp.quniform('eta', 0.025, 0.5, 0.025),
             'max_depth' : hp.quniform('max_depth', 1, 13, 1),
             'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
             'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
             'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
             'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
             'eval_metric': 'rmse',
             'objective': 'reg:linear',
             'silent' : 1
             } 
             
     
   
   
#######################################################################################


    

##################### read the train data
train_data =pd.read_csv('/media/shashank/Data/Projects/Kaggle/Home Depot/train.csv',\
                        encoding = "ISO-8859-1")

##################### read the product descriptions data
proddesc_data =pd.read_csv('/media/shashank/Data/Projects/Kaggle/Home Depot/product_descriptions.csv',\
                        encoding = "ISO-8859-1")
                      
##################### read the attribute  data
attr_data =pd.read_csv('/media/shashank/Data/Projects/Kaggle/Home Depot/attributes.csv'\,
                       encoding = "ISO-8859-1")


##################### read the attribute data frame
attr_dfcsv =pd.read_csv('/media/shashank/Data/Projects/Kaggle/Home Depot/data/attr_df.csv',\
                        encoding = "ISO-8859-1")
attr_dfcsv =attr_dfcsv.drop(['Unnamed: 0'],axis=1)


##################### read the test  data
X_test_df =pd.read_csv('/media/shashank/Data/Projects/Kaggle/Home Depot/data/X_test_df.csv')
X_test_df = X_test_df.drop(['Unnamed: 0'],axis=1)
X_test_df.columns



#################### merge the product description and train data ( left outer join)
train_proddesc = pd.merge(left=proddesc_data,right=train_data,left_on='product_uid',\
                          right_on='product_uid')



################### merge the train and attribue
train_attr = pd.merge(left=train_data,right=attr_dfcsv,how='left',left_on='product_uid',\
                          right_on='product_uid')
                          

##################### modify the attribute data 
columns2 = ["product_uid" ,"value"]
attr_df =  pd.DataFrame({"product_uid" :[100001],
                         "value" : ["Versatile connector for various 90Â° connections and home repair projects"]
                       },   
                        columns=columns2)

temp_str = str(attr_data["value"][0])                       
k = 0                        
j = attr_data['product_uid'][0]
for i in range(1,len(attr_data)):
  if(math.isnan(attr_data['product_uid'][i]) == False):  
     if(attr_data['product_uid'][i] == j): 
        temp_str = " ".join([temp_str , str(attr_data["value"][i]) ])
     else:
       attr_df.iat[k,1] = temp_str  
       temp_str = str(attr_data["value"][i])
       temp = pd.DataFrame({"product_uid" :[ attr_data['product_uid'][i]],
                         "value" : [ attr_data["value"][i]]
                       },   
                        columns=columns2)   
       attr_df = pd.concat( [attr_df,temp] , axis=0)  
       j = attr_data['product_uid'][i]
       k = k+1


##################### modify attr_df and train_data
attr_df = attr_df.set_index('product_uid')
sno = range(0,attr_df.shape[0])
attr_df['sno'] = sno

train_data2 = train_data
train_data2 = train_data2.set_index('product_uid')
sno1 = range(0,len(train_data2))
train_data2['sno']=sno1


###################### clean the product titles and search queries 
clean_alldata =[]
 
for i in range( 0, len(train_data) ):
     clean_alldata.append( process_strings(str(train_data["product_title"][i]) ) )

for i in range( 0, len(train_data) ):
     clean_alldata.append( process_strings(str(train_data["search_term"][i]) ) )

for i in range( 0, len(proddesc_data)):
     clean_alldata.append( process_strings( str(proddesc_data["product_description"][i]) ) ) 
     
for i in range( 0, len(attr_df)):
     clean_alldata.append( process_strings(attr_dfcsv["value"][i]))    



####### create tf-idf vectors for cleaned  product titles , desc and search queries  
vectorizer1 = TfidfVectorizer(analyzer = "word",  token_pattern = r"\b[a-z0-9]+\b",\
               binary = True,norm = 'l2', )
tfidf_features = vectorizer1.fit_transform(clean_alldata)
tfidf = vectorizer1.idf_
tfidf_dict = dict(zip(vectorizer1.get_feature_names(), tfidf))


###### isolate the if-idf features of the  product titles , desc and search queries
tfidf_features_prodc_title = tfidf_features[0:len(train_data)]
tfidf_features_search_term = tfidf_features[len(train_data):2*len(train_data)]
tfidf_features_prodc_desc = tfidf_features[2*len(train_data):2*len(train_data)+len(proddesc_data)]
tfidf_features_attr = tfidf_features[2*len(train_data)+len(proddesc_data):2*len(train_data)\
                                            +len(proddesc_data)+len(attr_df)]

tfidf_feature_names = vectorizer1.get_feature_names()
tfidf_feature_names[23444:23499]
 
 
    
######### create a binary bag of words for cleaned  product titles , desc and search queries
vectorizer2 = CountVectorizer(analyzer = "word", token_pattern = r"\b[a-z0-9]+\b",\
                            binary = True,preprocessor = None,stop_words = None)
bintf_features = vectorizer2.fit_transform(clean_alldata)
feature_names = vectorizer2.get_feature_names()
feature_names[23444:23499]


###### isolate the binary bag of words of cleaned  product titles , desc and search queries
bintf_features_prodc_title = bintf_features[0:len(train_data)]
bintf_features_search_term = bintf_features[len(train_data):2*len(train_data)]
bintf_features_prodc_desc = bintf_features[2*len(train_data):2*len(train_data)+len(proddesc_data)]
bintf_features_attr = bintf_features[2*len(train_data)+len(proddesc_data):2*len(train_data)\
                                            +len(proddesc_data)+len(attr_df)]


######### create a freq bag of words for cleaned  product titles , desc and search queries
vectorizer3 = CountVectorizer(analyzer = "word", token_pattern = r"\b[a-z0-9]+\b",binary = False,\
              preprocessor = None,stop_words = None)
freq_features = vectorizer3.fit_transform(clean_alldata)



###### isolate the freq bag of words of cleaned  product titles , desc and search queries
freq_features_prodc_title = freq_features[0:len(train_data)]
freq_features_search_term = freq_features[len(train_data):2*len(train_data)]
freq_features_prodc_desc = freq_features[2*len(train_data):2*len(train_data)+len(proddesc_data)]
freq_features_attr = freq_features[2*len(train_data)+len(proddesc_data):2*len(train_data)\
                                            +len(proddesc_data)+len(attr_df)]

###### compute freq scores
freq_scores_title = np.zeros(shape=freq_features_prodc_title.shape[0])
for i in range(0,freq_features_prodc_title.shape[0]):
    x = bintf_features_search_term[i]
    y = freq_features_prodc_title[i]
    x = x.toarray()
    y = y.toarray()
    z = np.multiply(x,y)
    freq_scores_title[i] = np.sum(z)


freq_scores_desc = np.zeros(shape=freq_features_prodc_title.shape[0])
for i in range(0,freq_features_prodc_title.shape[0]):
    x = bintf_features_search_term[i]
    y = freq_features_prodc_desc[train_data['product_uid'][i]-100001]
    x = x.toarray()
    y = y.toarray()
    z = np.multiply(x,y)
    freq_scores_desc[i] = np.sum(z)


freq_scores_attr = np.zeros(shape=freq_features_prodc_title.shape[0])
for i in range(0,freq_features_prodc_title.shape[0]):
    x = bintf_features_search_term[i]
    if (pd.isnull(train_attr['value'][i])):
      freq_scores_attr[i] = -999
    else:
      puid = train_attr['product_uid'][i]
      ind = attr_df['sno'][puid]
      y = freq_features_attr[ind]
      x = x.toarray()
      y = y.toarray()
      z = np.multiply(x,y)
      freq_scores_attr[i] = np.sum(z)




###### compute tf-idf scores
tf_idf_scores_title = np.zeros(shape=tfidf_features_prodc_title.shape[0])
for i in range(0,tfidf_features_prodc_title.shape[0]):
    x = bintf_features_search_term[i]
    y = tfidf_features_prodc_title[i]
    x = x.toarray()
    y = y.toarray()
    z = np.multiply(x,y)
    tf_idf_scores_title[i] = np.sum(z)


tf_idf_scores_desc = np.zeros(shape=tfidf_features_prodc_title.shape[0])
for i in range(0,tfidf_features_prodc_title.shape[0]):
    x = bintf_features_search_term[i]
    y = tfidf_features_prodc_desc[train_data['product_uid'][i]-100001]
    x = x.toarray()
    y = y.toarray()
    z = np.multiply(x,y)
    tf_idf_scores_desc[i] = np.sum(z)


tf_idf_scores_attr = np.zeros(shape=freq_features_prodc_title.shape[0])
for i in range(0,freq_features_prodc_title.shape[0]):
    x = bintf_features_search_term[i]
    if (pd.isnull(train_attr['value'][i])):
      tf_idf_scores_attr[i] = -999
    else:
      puid = train_attr['product_uid'][i]
      ind = attr_df['sno'][puid]
      y = tfidf_features_attr[ind]
      x = x.toarray()
      y = y.toarray()
      z = np.multiply(x,y)
      tf_idf_scores_attr[i] = np.sum(z)




###### compute the sizes 
query_size = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = bintf_features_search_term[i]
    x = x.toarray()
    query_size[i] = np.sum(x)

title_size = np.zeros(shape=bintf_features_prodc_title.shape[0])
for i in range(0,bintf_features_prodc_title.shape[0]):
    x = bintf_features_prodc_title[i]
    x = x.toarray()
    title_size[i] = np.sum(x)
    
desc_size = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = bintf_features_prodc_desc[train_data['product_uid'][i]-100001]
    x = x.toarray()
    desc_size[i] = np.sum(x)

attr_size = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    if (pd.isnull(train_attr['value'][i])):
      attr_size[i] = -999
    else:
      puid = train_attr['product_uid'][i]
      ind = attr_df['sno'][puid]
      x= bintf_features_attr[ind]  
      x = x.toarray()
      attr_size[i] = np.sum(x)


###### compute the number of terms matching
match_title = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = bintf_features_search_term[i]
    y = bintf_features_prodc_title[i]
    x = x.toarray()
    y = y.toarray()
    z = np.multiply(x,y)
    match_title[i] = np.sum(z)


match_desc = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = bintf_features_search_term[i]
    y = bintf_features_prodc_desc[train_data['product_uid'][i]-100001]
    x = x.toarray()
    y = y.toarray()
    z = np.multiply(x,y)
    match_desc[i] = np.sum(z)

match_attr = np.zeros(shape=freq_features_prodc_title.shape[0])
for i in range(0,freq_features_prodc_title.shape[0]):
    x = bintf_features_search_term[i]
    if (pd.isnull(train_attr['value'][i])):
      match_attr[i] = -999
    else:
      puid = train_attr['product_uid'][i]
      ind = attr_df['sno'][puid]
      y = bintf_features_attr[ind]
      x = x.toarray()
      y = y.toarray()
      z = np.multiply(x,y)
      match_attr[i] = np.sum(z)




####### compute ratio features- tfidf weight with match criteria
feat_title = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = tfidf_features_search_term[i]
    y = bintf_features_prodc_title[i]
    x = x.toarray()
    y = y.toarray()
    z = np.multiply(x,y)
    num = np.sum(z)
    den = np.sum(x)
    feat_title[i] = num/den


feat_desc = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = tfidf_features_search_term[i]
    y = bintf_features_prodc_desc[train_data['product_uid'][i]-100001]
    x = x.toarray()
    y = y.toarray()
    z = np.multiply(x,y)
    num = np.sum(z)
    den = np.sum(x)
    feat_desc[i] = num/den


feat_attr = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = tfidf_features_search_term[i]
    if (pd.isnull(train_attr['value'][i])):
      feat_attr[i] = 999
    else:
      puid = train_attr['product_uid'][i]
      ind = attr_df['sno'][puid]
      y = bintf_features_attr[ind]
      x = x.toarray()
      y = y.toarray()
      z = np.multiply(x,y)
      num = np.sum(z)
      den = np.sum(x)
      feat_attr[i] = num/den



####### compute a feature with match ratio criteria
feat_title1 = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = bintf_features_search_term[i]
    y = bintf_features_prodc_title[i]
    x = x.toarray()
    y = y.toarray()
    z = np.multiply(x,y)
    num = np.sum(z)
    den = np.sum(x)
    feat_title1[i] = num/den


feat_desc1 = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = bintf_features_search_term[i]
    y = bintf_features_prodc_desc[train_data['product_uid'][i]-100001]
    x = x.toarray()
    y = y.toarray()
    z = np.multiply(x,y)
    num = np.sum(z)
    den = np.sum(x)
    feat_desc1[i] = num/den


feat_attr1 = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = bintf_features_search_term[i]
    if (pd.isnull(train_attr['value'][i])):
      feat_attr1[i] = -999
    else:
      puid = train_attr['product_uid'][i]
      ind = attr_df['sno'][puid]
      y = bintf_features_attr[ind]
      x = x.toarray()
      y = y.toarray()
      z = np.multiply(x,y)
      num = np.sum(z)
      den = np.sum(x)
      feat_attr1[i] = num/den
      
      

################## BM25 distance ##################
BM25_title = np.zeros(shape=bintf_features_search_term.shape[0])
title_avg = title_size.mean()
for i in range(0, len(freq_scores_title)):
        abs_len = title_size[i].sum()
        f = freq_scores_title[i]
        idf = feat_title[i]
        k = 1.5
        b = 0.75
        bm = idf * (f*(k + 1) / (f + k*(1 - b + b*abs_len/title_avg)))
        BM25_title[i] = bm
        

BM25_desc = np.zeros(shape=bintf_features_search_term.shape[0])
desc_avg = desc_size.mean()    
for i in range(0, len(freq_scores_desc)):
        abs_len = desc_size[i].sum()
        f = freq_scores_desc[i]
        idf = feat_desc[i]
        k = 1.5
        b = 0.75
        bm = idf * (f*(k + 1) / (f + k*(1 - b + b*abs_len/desc_avg)))
        BM25_desc[i] = bm  


BM25_attr = np.zeros(shape=bintf_features_search_term.shape[0])
attr_avg = attr_size.mean()    
for i in range(0, len(freq_scores_attr)):
     if (pd.isnull(train_attr['value'][i])): 
        BM25_attr[i] = -999
     else:
        abs_len = attr_size[i].sum()
        f = freq_scores_attr[i]
        idf = feat_attr[i]
        k = 1.5
        b = 0.75
        bm = idf * (f*(k + 1) / (f + k*(1 - b + b*abs_len/attr_avg)))
        BM25_attr[i] = bm  
        
        

####### Jaccard's coefficient - binary weights
jacc_coeff_title = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = bintf_features_search_term[i]
    y = bintf_features_prodc_title[i]
    x = x.toarray()
    y = y.toarray()
    jacc_coeff_title[i] = metrics.jaccard_similarity_score(x,y,sample_weight=None)

jacc_coeff_desc = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = bintf_features_search_term[i]
    y = bintf_features_prodc_desc[train_data['product_uid'][i]-100001]
    x = x.toarray()
    y = y.toarray()
    jacc_coeff_desc[i] = metrics.jaccard_similarity_score(x,y,sample_weight=None)

jacc_coeff_attr = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = bintf_features_search_term[i]
    if (pd.isnull(train_attr['value'][i])):
      jacc_coeff_attr[i] = -999
    else:
      puid = train_attr['product_uid'][i]
      ind = attr_df['sno'][puid]
      y = bintf_features_attr[ind]
      x = x.toarray()
      y = y.toarray()
      jacc_coeff_attr[i] = metrics.jaccard_similarity_score(x,y,sample_weight=None)



###### simple matching ratios for title and description
match_ratio_title = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = bintf_features_search_term[i]
    y = bintf_features_prodc_title[i]
    x = x.toarray()
    y = y.toarray()
    z = np.multiply(x,y)
    match_ratio_title[i] = np.sum(z)/np.sum(x)


match_ratio_desc = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = bintf_features_search_term[i]
    y = bintf_features_prodc_desc[train_data['product_uid'][i]-100001]
    x = x.toarray()
    y = y.toarray()
    z = np.multiply(x,y)
    match_ratio_desc[i] = np.sum(z)/np.sum(x)

match_ratio_attr = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = bintf_features_search_term[i]
    if (pd.isnull(train_attr['value'][i])):
      match_ratio_attr[i] = -999
    else:
      puid = train_attr['product_uid'][i]
      ind = attr_df['sno'][puid]
      y = bintf_features_attr[ind]
      x = x.toarray()
      y = y.toarray()
      match_ratio_attr[i] = np.sum(z)/np.sum(x)



######## cosine similarity - matching
cosine_sim_title = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    xb = bintf_features_search_term[i]
    xt = tfidf_features_search_term[i]
    y = tfidf_features_prodc_title[i]
    xb = xb.toarray()
    xt = xt.toarray()
    y = y.toarray()
    ym = np.multiply(y,xb)
    cosine_sim_title[i] = cosine_similarity(xt,ym)

cosine_sim_desc = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    xb = bintf_features_search_term[i]
    xt = tfidf_features_search_term[i]
    y = tfidf_features_prodc_desc[train_data['product_uid'][i]-100001]
    xb = xb.toarray()
    xt = xt.toarray()
    y = y.toarray()
    ym = np.multiply(y,xb)
    cosine_sim_desc[i] = cosine_similarity(xt,ym)
    
    
cosine_sim_attr = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = bintf_features_search_term[i]
    if (pd.isnull(train_attr['value'][i])):
      cosine_sim_attr[i] = -999
    else:
      xb = bintf_features_search_term[i]
      xt = tfidf_features_search_term[i]
      puid = train_attr['product_uid'][i]
      ind = attr_df['sno'][puid]
      y = tfidf_features_attr[ind]
      xb = xb.toarray()
      xt = xt.toarray()
      y = y.toarray()
      ym = np.multiply(y,xb)
      cosine_sim_attr[i] = cosine_similarity(xt,ym)



######## cosine similarity - complete
cosine_sim_title_comp = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = tfidf_features_search_term[i]
    y = tfidf_features_prodc_title[i]
    x = x.toarray()
    y = y.toarray()
    cosine_sim_title_comp[i] = cosine_similarity(x,y)

cosine_sim_desc_comp = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = tfidf_features_search_term[i]
    y = tfidf_features_prodc_desc[train_data['product_uid'][i]-100001]
    x = x.toarray()
    y = y.toarray()
    cosine_sim_desc_comp[i] = cosine_similarity(x,y)
    

cosine_sim_attr_comp = np.zeros(shape=bintf_features_search_term.shape[0])
for i in range(0,bintf_features_search_term.shape[0]):
    x = tfidf_features_search_term[i]
    if (pd.isnull(train_attr['value'][i])):
      cosine_sim_attr_comp[i] = -999
    else:
      puid = train_attr['product_uid'][i]
      ind = attr_df['sno'][puid]
      y = tfidf_features_attr[ind]
      x = x.toarray()
      y = y.toarray()
      cosine_sim_attr_comp[i] = cosine_similarity(x,y)    
      


     
  

############## create dataframe for training 
columns = ['query_size','title_size','desc_size','attr_size','match_ratio_title','match_ratio_desc',\
           'match_ratio_attr','tf_idf_scores_title','tf_idf_scores_desc','tf_idf_scores_attr',\
           'feat_title','feat_desc','feat_attr','feat_title1','feat_desc1','feat_attr1',\
           'jacc_coeff_title','jacc_coeff_desc','jacc_coeff_attr','cosine_sim_title','cosine_sim_desc',\
           'cosine_sim_attr','cosine_sim_title_comp','cosine_sim_desc_comp','cosine_sim_attr_comp',\
            'BM25_title','BM25_desc','BM25_attr']
X_df = pd.DataFrame(
    {'query_size': query_size,
     'title_size': title_size,
     'desc_size':desc_size,
     'attr_size': attr_size,
     'match_ratio_title': match_ratio_title,
     'match_ratio_desc': match_ratio_desc,
     'match_ratio_attr': match_ratio_attr,
     'tf_idf_scores_title' : tf_idf_scores_title,
     'tf_idf_scores_desc': tf_idf_scores_desc,
     'tf_idf_scores_attr': tf_idf_scores_attr,
     'feat_title': feat_title,
     'feat_desc': feat_desc,
     'feat_attr': feat_attr,
     'feat_title1': feat_title1,
     'feat_desc1': feat_desc1,
     'feat_attr1': feat_attr1,     
     'jacc_coeff_desc':jacc_coeff_desc,
     'jacc_coeff_title':jacc_coeff_title,
     'jacc_coeff_attr':jacc_coeff_attr,
     'cosine_sim_title':cosine_sim_title,
     'cosine_sim_desc':cosine_sim_desc,
     'cosine_sim_attr':cosine_sim_attr,
     'cosine_sim_title_comp':cosine_sim_title_comp,
     'cosine_sim_desc_comp':cosine_sim_desc_comp,
     'cosine_sim_attr_comp':cosine_sim_attr_comp,
     'BM25_title' : BM25_title,
     'BM25_desc' : BM25_desc,
     'BM25_attr' : BM25_attr,
      }, columns=columns) 

Y = train_data["relevance"]


####### create validation and training sets
X_train,X_val,Y_train,Y_val = train_test_split(X_df,Y,test_size=0.2)


############################### XGBoost regression #######################################
Y_train= np.ravel(Y_train)
Y_val= np.ravel(Y_val)

params = {
         'max_depth': 3,
         'learning_rate=0.05': 0.05
         }


num_round = 1000
dtrain = xgb.DMatrix(X_train, label=Y_train)
dvalid = xgb.DMatrix(X_val, label=Y_val)
model = xgb.train(params, dtrain, num_round)
predictions = model.predict(dvalid)

out_sample_rmse_xgboost = mean_squared_error(Y_val, predictions)**0.5 

###########################################################################################



######################## Hyperparameter tuning using Hyperopot - XGBoost #######################
best = fmin(score, space, algo=tpe.suggest,max_evals=250)
print(best)

################################################################################################




############################ Random Forest Regressor #####################################    
Y_train= np.ravel(Y_train)
Y_val= np.ravel(Y_val)

model_rf_reg = RandomForestRegressor(n_estimators=1000,max_features='sqrt',oob_score=True)
model_rf_reg.fit(X_train, Y_train)

Y_pred = model_rf_reg.predict(X_train)
in_sample_rmse = mean_squared_error(Y_train, Y_pred)**0.5 

Y_predval = model_rf_reg.predict(X_val)
Y_predtrain = model_rf_reg.predict(X_train)
out_sample_rmse = mean_squared_error(Y_val, Y_predval)**0.5 


###### feature importance
model_rf_reg.feature_importances_
model_rf_reg.oob_score_

#####################################################################################################


############################ Random Forest Regressor - Complete Data ###############################    
Y= np.ravel(Y)


model_rf_reg_full = RandomForestRegressor(n_estimators=1200,max_features='sqrt',oob_score=True)
model_rf_reg_full.fit(X_df, Y)


###### feature importance
model_rf_reg_full.feature_importances_
model_rf_reg_full.oob_score_

#####################################################################################################


############################ Prediction on the test data ####################################
pred_test_df = model_rf_reg_full.predict(X_test_df)

##############################################################################################









###### export result to csv

prob_df.to_csv('prob_compare.csv')
prob_iso_df.to_csv('prob_iso_compare.csv')
prob_train_df.to_csv('prob_train_df.csv')
val_df.to_csv('val_df.csv')


X_df.to_csv('X_df.csv')
Y.to_csv('Y_df.csv')

pred_test_df = pd.DataFrame(pred_test_df)
pred_test_df.to_csv('pred_test_df.csv')

attr_df.to_csv('attr_df.csv')




















          