from django.shortcuts import render
from scipy import sparse
import pickle
import pandas as pd
from sklearn.metrics import pairwise_distances
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from random import randint
import math
import time
import re
import os
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
from matplotlib import gridspec
from scipy.sparse import hstack

data = pd.read_pickle('16k_apperal_data_preprocessed').reset_index()



def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    return Counter(words) 

def get_result(doc_id, content_a, content_b, url, model):
    text1 = content_a
    text2 = content_b
    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)

#BOW
title_features = sparse.load_npz("bow_feat.npz")
img_url_bow = []
def bag_of_words_model(doc_id, num_results):
    pairwise_dist = pairwise_distances(title_features,title_features[doc_id])
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
    df_indices = list(data.index[indices])
    for i in range(0,len(indices)):
        get_result(indices[i],data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], 'bag_of_words')
        img_url_bow.append(data['medium_image_url'].loc[df_indices[i]])
bag_of_words_model(12566, 10)


#TFIDF
tfidf_title_features = sparse.load_npz("tfidf_feat.npz")
img_url_tfidf = []
def tfidf_model(doc_id, num_results):
    pairwise_dist = pairwise_distances(tfidf_title_features,tfidf_title_features[doc_id])
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
    df_indices = list(data.index[indices])
    for i in range(0,len(indices)):
        get_result(indices[i],data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], 'bag_of_words')
        img_url_tfidf.append(data['medium_image_url'].loc[df_indices[i]])
tfidf_model(12566, 10)


def home(request):
    q_img = data.medium_image_url[12566]
    result_url_tfidf = img_url_tfidf
    result_url_bow = img_url_bow
    return render(request, 'fashion_engine/home.html',{'q_img':q_img,'img_url_bow':result_url_bow,'img_url_tfidf':result_url_tfidf})



def select_img(request):
    rand_img = []
    idx = []
    for i in range(20):
        k = randint(0, 16042)
        idx.append(k)
        rand_img.append(data.medium_image_url[k])
    images = rand_img
    idx_img = idx 
    rand_img = []
    idx = []
    print(request.GET) 
    return render(request, 'fashion_engine/select_img.html',{'images':images,'idx_img':idx_img})

