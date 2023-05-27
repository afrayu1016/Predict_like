
import pandas as pd
import tensorflow as tf
import string , re
import numpy as np
from datetime import datetime, timezone
import statistics
from statistics import mean 
import matplotlib.pyplot as plt
import copy
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import joblib
from collections import Counter

import gensim
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec

from gensim import models
from sklearn.model_selection import train_test_split 

from sklearn import preprocessing

from collections import Counter

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
from sklearn.svm import SVR




## reomove mark
def remove_mark (df):    
    
    title_arr = []

    for item in df.title:
        s = item
        s = re.sub(r'[^\w\s]','',s)
        title_arr.append(s)
        
    return title_arr


## ckiptagger
## return splited sentence

def sen_list (df):
    # package of ckiptagger
    ws = WS("./data")
    pos = POS("./data")
    ner = NER("./data")

    sentence_list = remove_mark(df)

    word_sentence_list = ws(
        sentence_list,
        sentence_segmentation = True, 
        )
    pos_sentence_list = pos(word_sentence_list)
    
    return word_sentence_list, pos_sentence_list


def sen2dic(df):
    
    word_sentence_list , pos_sentence_list = sen_list(df)
    
    word_dic = {}
    pos_dic = {}
    for i, sentence in enumerate(remove_mark(df)):
        name =  str(i)
        word_dic.update({name : word_sentence_list[i]})
        pos_dic.update({name: pos_sentence_list[i]})
        
    return word_dic, pos_dic



## remove important word by pos_dic
def re_unimportance(df):
    
    word_dic, pos_dic = sen2dic (df)
    
    remove_word_dic = {}
    remove_tag = [
                'Cab','Cba','Cbb',
                'D','Da','Dfa','Dfb','Di','Dk',
                'I','Nf','Neqa','Neu',
                'P','T',
                'Nh','Nep',
                'DE',
                'SHI',
                'WHITESPACE',
                'SPCHANGECATEGORY',
                'SEMICOLONCATEGORY',
                'QUESTIONCATEGORY',
                'PERIODCATEGORY',
                'PAUSECATEGORY',
                'PARENTHESISCATEGORY',
                'EXCLAMATIONCATEGORY',
                'ETCCATEGORY',
                'DOTCATEGORY',
                'DASHCATEGORY',
                'COMMACATEGORY',
                'COLONCATEGORY']

    for title in word_dic.keys():

        item = []

        for i , tag in enumerate(pos_dic.get(title)):

            t = False

            for del_tag in remove_tag:
                if tag == del_tag:
                    t = True
                    break

            ## remove unimportant word by tag & remove icon(not words)
            if t == False and word_dic.get(title)[i].strip().isalpha():
                item.append(word_dic.get(title)[i].strip())

        remove_word_dic.update({title : item})
        
    return remove_word_dic


## transfer the title dealt by ckiptagger into dataframe
def dic2df(df):
    remove_word_dic = re_unimportance(df)
    word_pd = pd.DataFrame(index = remove_word_dic.keys(),columns = ['Word_split_arr'])
    for title in remove_word_dic.keys():
        word_pd.at[title,'Word_split_arr'] = remove_word_dic.get(title)
    return word_pd


### Word2Vec

def w2v (df_train, df_test, df_private) :
    
    train_word_pd = dic2df(df_train)
    test_word_pd = dic2df(df_test)
    private_word_pd = dic2df(df_private)
    
    word_pd = train_word_pd.append (test_word_pd)
    
    # Settings
    seed = 666
    sg = 0
    window_size = 10
    vector_size = 100
    min_count = 1
    workers = 8
    epochs = 5
    batch_words = 10000

    train_data = word_pd['Word_split_arr'] 

    model = word2vec.Word2Vec(
        train_data,
        min_count=min_count,
        vector_size=vector_size,
        workers=workers,
        epochs=epochs,
        window=window_size,
        sg=sg,
        seed=seed,
        batch_words=batch_words
    )

    model.save('./word2vec.model')
    model = word2vec.Word2Vec.load('./word2vec.model')
    words = set(model.wv.index_to_key) 
    
    X_train_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) 
                         for ls in train_word_pd['Word_split_arr']]) 
    
    X_test_vect = np.array( [np.array([model.wv[i] for i in ls if i in words]) 
                         for ls in test_word_pd['Word_split_arr']])
    
    X_private_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) 
                         for ls in private_word_pd['Word_split_arr']]) 
    
    return X_train_vect, X_test_vect, X_private_vect


# average the vector of each title, turn each vector length into the same length

def avg_vector(data_vector):
    
    vect_avg = [] 
    for v in data_vector: 
        if v.size: 
            vect_avg.append(v.mean(axis=0)) 
        else: 
            vect_avg.append(np.zeros(100, dtype=float)) 
            
    return vect_avg


### Transfer the time to hour

def extract_h(df):
    
    time_arr = []
    
    for time in df.created_at:

        creared_time = time
        creared_time = creared_time.replace('UTC','') ## time'2022-09-13 03:25:47 '
        created_hour = creared_time.split(' ')[1].split(':')[0] ## extract hour
        created_hour = int(created_hour)
        time_arr.append(created_hour)
        
    return time_arr


###  Avarege increment of comment count and like count

def avg_incre(df, type_hr):
    
    avg_incre_arr = []
    final_hr = 5
    
    for index in range(0,len(df.index)):
        final_count = df.at[index, type_hr[final_hr]]
        avg_incre = final_count/6
        avg_incre_arr.append(avg_incre)
        
    return avg_incre_arr



### Transfer label data 
## 1 : 0-50, 2 : 51-99, 3 : 100-149, 4: 150-199, 5: 200-299, 6:300-399 ...
def _transfer_count (label_data):
    transfer_count = []
    count_slice = 50
    
    for like_count in label_data.to_numpy():
        
        temp = int(like_count / 10)
        
        if like_count >=0 and like_count < 50:
            transfer_count.append(1)
            continue
        elif like_count >= 50 and like_count < 100:
            transfer_count.append(2)
            continue
        elif like_count >=100 and like_count < 150:
            transfer_count.append(3)
            continue
        elif like_count >=150 and like_count < 200:
            transfer_count.append(4)
            continue
       

        temp = int(like_count / count_slice ) + 4
        transfer_count.append(temp)
    return transfer_count


def rf_data(df , title_w2v , word_cloud_freq, outlier_word_count):
    
    like_hr = ['like_count_1h','like_count_2h','like_count_3h',
               'like_count_4h','like_count_5h','like_count_6h']
    
    comment_hr = ['comment_count_1h','comment_count_2h','comment_count_3h',
                  'comment_count_4h','comment_count_5h','comment_count_6h']
    
    rf_data = pd.DataFrame()

    rf_data = rf_data.assign(extract_h = extract_h(df))
    rf_data = rf_data.assign(like_avg_incre = avg_incre(df,like_hr))
    rf_data = rf_data.assign(comment_avg_incre = avg_incre(df,comment_hr))

    rf_data = pd.concat([rf_data, title_w2v, word_cloud_freq, outlier_word_count], axis =1)
    rf_data = rf_data.assign(like_count_24h = df['like_count_24h'])
    rf_data = rf_data.assign(like_count_24h_label = _transfer_count(df['like_count_24h']))
    
    return rf_data
    


def rf_private_data(df , title_w2v , word_cloud_freq, outlier_word_count):
    
    like_hr = ['like_count_1h','like_count_2h','like_count_3h',
               'like_count_4h','like_count_5h','like_count_6h']
    
    comment_hr = ['comment_count_1h','comment_count_2h','comment_count_3h',
                  'comment_count_4h','comment_count_5h','comment_count_6h']
    
    rf_data = pd.DataFrame()

    rf_data = rf_data.assign(extract_h = extract_h(df))
    rf_data = rf_data.assign(like_avg_incre = avg_incre(df,like_hr))
    rf_data = rf_data.assign(comment_avg_incre = avg_incre(df,comment_hr))

    # without like_count_24h_label
    rf_data = pd.concat([rf_data, title_w2v, word_cloud_freq, outlier_word_count], axis =1)
    
    
    return rf_data


def sample_data (original_data, split_percent):
    _shffuled = original_data.sample(n = split_percent, axis = 0)
    remove_shuffle = original_data[~original_data.isin(_shffuled)].dropna()
    return _shffuled, remove_shuffle



def rf_para():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
#     print(random_grid)

    return random_grid
#     {'bootstrap': [True, False],
#      'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#      'max_features': ['auto', 'sqrt'],
#      'min_samples_leaf': [1, 2, 4],
#      'min_samples_split': [2, 5, 10],
#      'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}



### model - RandomForest 
def rf_model (dataset , X_test, test_label, rf_private):
    
    rf = RandomForestClassifier(bootstrap = True,
                                max_depth = 90,
                                min_samples_leaf = 2,
                                min_samples_split = 12, 
                                n_estimators = 400)
    
    
    y_train = dataset['like_count_24h_label']
    X_train = dataset.drop(['like_count_24h_label','like_count_24h'], axis =1)
    
    rf_model = rf.fit(X_train , y_train)
    pred = rf_model.predict(X_test)
    private_label = rf_model.predict(rf_private)## private
    
    ## store the training and testing result predicted by random forest 
    train_pred = pd.DataFrame(rf_model.predict(X_train), columns = ['rf_label'] , index = dataset.index) ## rf for training set
    test_pred = pd.DataFrame(pred, columns = ['rf_label']) 
    private_pred = pd.DataFrame(private_label, columns = ['rf_label']) 
    
    svm_data = pd.concat([dataset, train_pred], axis = 1)
    svm_test_data = pd.concat([X_test, test_pred], axis = 1)
    svm_private_data = pd.concat([rf_private, private_pred], axis =1)## private
    
    ## the performance of the random forest
    precision = precision_score(test_label, pred,  average = 'macro').round(4)
    recall = recall_score(test_label, pred,  average = 'macro').round(4)
    f1 =( 2 / ( (1/ precision) + (1/ recall))).round(4)
    
    return  svm_data , svm_test_data , svm_private_data, precision, recall, f1 



## Defining MAPE function
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape



def flat_arr (svm_data_arr):
    ttt = svm_data_arr[0]
    # ttt[0][:-1]
    svm_data_arr = [list(ttt[0][:-1]) + [i[3] for i in ttt] for ttt in svm_data_arr]
    return svm_data_arr

## remove extreme data
def outlier(df):
    df_med = np.median(df['like_count_24h'])
    df_std = df['like_count_24h'].std()
    
    _outlier = df_med + df_std*5
    
    new_df = df.drop(df[df['like_count_24h'] > _outlier].index ).reset_index(drop = True)
    outlier_df = df.drop(df[df['like_count_24h'] < _outlier].index ).reset_index(drop = True)
    
    return new_df, outlier_df

## count word frequency
def words_freq (word_pd , t):
    
    total = []
    for item in t:
        for word in item:
            total.append(total)
    all_word_len = len(total)
    all_word_len
    
    words_count = Counter(t) 
    words_sen_count = []
    for item in word_pd['Word_split_arr']:
        freqs = 0
        for words in item:
            wc = words_count.get(words)
            if wc is None:
                wc = 0
            freqs +=wc
        words_sen_count.append(freqs) ##/all_word_len
        
    normalized_arr = preprocessing.normalize(np.array([words_sen_count]))

    word_cloud_freq = pd.DataFrame(normalized_arr[0], columns = ['word_cloud_freq'])

    return word_cloud_freq


## count the words shown on outlier article is shown in training set, testing set, private set
def outlier_word_count(word_pd, outlier_t):
    
    outlier_words_count = Counter(outlier_t)

    outlier_words_sen_count = []

    for item in word_pd['Word_split_arr']:
        _count = 0
        for words in item:
            wc = outlier_words_count.get(words)
            if wc is None:
                continue
            else:
                _count +=1
        outlier_words_sen_count.append(_count)

    outlier_word_cloud_freq = pd.DataFrame(outlier_words_sen_count, columns = ['outlier_word_count'])
    return outlier_word_cloud_freq


###training model
## loading data 

df_train = pd.read_csv('./dataset/intern_homework_train_dataset.csv')
df_test = pd.read_csv('./dataset/intern_homework_public_test_dataset.csv')
df_private = pd.read_csv('./dataset/intern_homework_private_test_dataset.csv')

##outlier->remove the extreme data
df_train , outlier_df_train = outlier(df_train)
df_test ,outlier_df_test = outlier(df_test)


## Word2Vec
X_train_vect, X_test_vect, X_private_vect = w2v(df_train, df_test, df_private) 
X_train_vect_avg = avg_vector(X_train_vect)
X_test_vect_avg = avg_vector(X_test_vect) 

X_private_vect_avg = avg_vector(X_private_vect) 

_col = []
for i in range(1, 101):
    _col.append('v_'+str(i))

train_w2v = pd.DataFrame(X_train_vect_avg, columns = _col)
test_w2v = pd.DataFrame(X_test_vect_avg, columns = _col)
private_w2v = pd.DataFrame(X_private_vect_avg, columns = _col)


## word cloud freq
train_word_pd = dic2df(df_train)
test_word_pd = dic2df(df_test)
private_word_pd = dic2df(df_private)
tt_word_pd = pd.concat([train_word_pd,test_word_pd])

t = []

for item in tt_word_pd['Word_split_arr']:
    for words in item:
        t.append(words)
words_count = Counter(t)

train_word_cloud_freq = words_freq(train_word_pd, t)
test_word_cloud_freq = words_freq(test_word_pd, t)
private_word_cloud_freq = words_freq(private_word_pd, t)


##count hot words frequency 

outlier_train_word_pd = dic2df(outlier_df_train)
outlier_test_word_pd = dic2df(outlier_df_test)
outlier_tt_pd = pd.concat([outlier_train_word_pd,outlier_test_word_pd])

outlier_t = []

for item in outlier_tt_pd['Word_split_arr']:
    for words in item:
        outlier_t.append(words)
        
train_outlier_word_count = outlier_word_count(train_word_pd, outlier_t)
test_outlier_word_count = outlier_word_count(test_word_pd, outlier_t)
private_outlier_word_count = outlier_word_count(private_word_pd, outlier_t)


## random forest training and testing data
rf_train = rf_data(df_train, train_w2v, train_word_cloud_freq, train_outlier_word_count)
rf_test = rf_data(df_test, test_w2v ,test_word_cloud_freq, test_outlier_word_count)

rf_private = rf_private_data(df_private, private_w2v, private_word_cloud_freq, private_outlier_word_count) ## without like_count_24h_label

## for rf_train data 
## imbalanced data ->split training data with transfer_label 1 into 5 set
## each training sample include label_1:8641  and not_label_1: 7671
split_data = rf_train.drop(rf_train[rf_train.like_count_24h_label > 1].index)
other_data = rf_train.drop(rf_train[rf_train.like_count_24h_label < 2].index)

split_count = int(len(split_data) / 5)

y_test = rf_test['like_count_24h_label']
X_test = rf_test.drop(['like_count_24h_label','like_count_24h'], axis =1)

_train = ['train1', 'train2', 'train3', 'train4', 'train5']
_test = ['test1', 'test2', 'test3', 'test4', 'test5']
_private = ['private1','private2','private3','private4','private5']

precision, recall, _f1  = [],[],[]

dataset = dict()
test_dataset = dict()
private_dataset = dict()



st = split_data.copy()

for  train, test, private in zip(_train, _test, _private):
    ## split data, shffule
    _sample , split_temp = sample_data(st , split_count)
    st = split_temp
    X_train = _sample.append(other_data).sample(frac=1)
    
    ## rf_model prediction
    svm_train_data, svm_test_data , svm_private_data, prec, rec, f1= rf_model(X_train, X_test, y_test, rf_private)
    
    
    dataset[train] = svm_train_data ##  5 set from rf_train
    test_dataset[test] = svm_test_data
    private_dataset[private] = svm_private_data
    
    precision.append(prec)
    recall.append(rec)
    _f1.append(f1)
    
    
# print("Random Forest\n")
# print("Precision array :", precision )
# print("----------------------------")
# print("Recall array :", recall )
# print("----------------------------")
# print("F1 Score array :", _f1 )



## SVM training data 
svm_input = []
svm_y_input = []

for idx in rf_train.index:
    
    arr = []
    if idx in other_data.index.values:
    # 為other data
        for train in _train:
            arr.append(dataset[train].loc[idx,['extract_h','like_avg_incre','comment_avg_incre','rf_label','word_cloud_freq','outlier_word_count']].values)
                            
    else:
        for train in _train:
            if idx in dataset[train].index.values:
                for i in range(5):
                    arr.append(dataset[train].loc[idx,['extract_h','like_avg_incre','comment_avg_incre','rf_label','word_cloud_freq','outlier_word_count']].values)
            
                
    svm_input.append(arr)
            
    svm_y_input.append(rf_train.loc[idx,['like_count_24h']].values[0])
    
svm_input = flat_arr(svm_input)


##  svm test data
svm_test_input = []
svm_test_y_input = []
for idx in rf_test.index: 
    arr = []
    for test in _test:
        arr.append(test_dataset[test].loc[idx,['extract_h','like_avg_incre','comment_avg_incre','rf_label','word_cloud_freq','outlier_word_count']].values)
    svm_test_input.append(arr)
    svm_test_y_input.append(rf_test.loc[idx,['like_count_24h']].values[0])
    
svm_test_input = flat_arr(svm_test_input)



##  svm private data
svm_private_input = []
for idx in rf_private.index:
    arr = []
    for private in _private:
        arr.append(private_dataset[private].loc[idx,['extract_h','like_avg_incre','comment_avg_incre','rf_label','word_cloud_freq','outlier_word_count']].values)
    svm_private_input.append(arr)
    
svm_private_input = flat_arr(svm_private_input)


## SVM model
svm_model = svm.SVR(kernel ='rbf', epsilon =0.2)
svm_model.fit(svm_input, svm_y_input)
svm_predict = svm_model.predict(svm_test_input)
## print("MAPE" , MAPE (svm_predict, svm_test_y_input))

result_predict = svm_model.predict(svm_private_input)
final_result = pd.DataFrame(result_predict, columns=['like_count_24h'])
final_result.to_csv('./_output/result.csv')  



