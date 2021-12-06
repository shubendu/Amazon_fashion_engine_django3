from django.shortcuts import render,redirect
from scipy import sparse
import pickle
import pandas as pd
from sklearn.metrics import pairwise_distances
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
import re
from collections import Counter


data = pd.read_pickle('pickles/2k_apperal_data_preprocessed').reset_index()
bottleneck_features_train = np.load('pickles/16k_data_cnn_features11.npy')
asins = np.load('pickles/16k_data_cnn_feature_asins11.npy')
asins = list(asins)
df_asins = list(data['asin'])



def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    return Counter(words) 

def get_result(doc_id, content_a, content_b, url, model):
    text1 = content_a
    text2 = content_b
    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)


def home(request):
    from random import randint
    rand_img = []
    idx=[]
    title = []
    brand = []
    color = []
    price = []
    for i in range(36):
        k = randint(0,2000)
        idx.append(k)
        rand_img.append(data.medium_image_url[k])
        title.append(data.title[k][:30]+'...')
        brand.append(data.brand[k])
        color.append(data.color[k])
        price.append(data.formatted_price[k])
    context1 = zip(rand_img,idx,title,brand,color,price)
    return render(request,'fashion_engine/home.html',{'context1':context1})


def output(request):
    print(request.GET.get('scale'))
    if  request.GET.get('scale') == None:
        return redirect('home')
    else:
        #BOW-------------------------------------------------------------------------------------------------------------------------
        title_features = sparse.load_npz("pickles/bow_feat2k.npz")
        img_url_bow = []
        title_bow = []
        color_bow = []
        brand_bow = []
        euc_dist_bow = []
        def bag_of_words_model(doc_id, num_results):
            pairwise_dist = pairwise_distances(title_features,title_features[doc_id])
            indices = np.argsort(pairwise_dist.flatten())[0:num_results]
            pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
            df_indices = list(data.index[indices])
            for i in range(0,len(indices)):
                get_result(indices[i],data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], 'bag_of_words')
                img_url_bow.append(data['medium_image_url'].loc[df_indices[i]])
                title_bow.append(data['title'].loc[df_indices[i]])
                color_bow.append(data['color'].loc[df_indices[i]])
                brand_bow.append(data['brand'].loc[df_indices[i]])
                euc_dist_bow.append(round(pdists[i],4))
        #BOW END--------------------------------------------------------------------------------------------------------------------------

        #TFIDF------------------------------------------------------------------------------------------------------------------------
        tfidf_title_features = sparse.load_npz("pickles/tfidf_feat2k.npz")
        img_url_tfidf = []
        title_tfidf = []
        color_tfidf = []
        brand_tfidf = []
        euc_dist_tfidf = []
        def tfidf_model(doc_id, num_results):
            pairwise_dist = pairwise_distances(tfidf_title_features,tfidf_title_features[doc_id])
            indices = np.argsort(pairwise_dist.flatten())[0:num_results]
            pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
            df_indices = list(data.index[indices])
            for i in range(0,len(indices)):
                get_result(indices[i],data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], 'bag_of_words')
                img_url_tfidf.append(data['medium_image_url'].loc[df_indices[i]])
                title_tfidf.append(data['title'].loc[df_indices[i]])
                color_tfidf.append(data['color'].loc[df_indices[i]])
                brand_tfidf.append(data['brand'].loc[df_indices[i]])
                euc_dist_tfidf.append(round(pdists[i],4))
        #TFIDF END-----------------------------------------------------------------------------------------------------------------------

        #CNN-----------------------------------------------------------------------------------------------------------------------------
        img_url_cnn = []
        title_cnn = []
        color_cnn = []
        brand_cnn = []
        euc_dist_cnn = []
        def get_similar_products_cnn(doc_id, num_results):
        
            doc_id = asins.index(df_asins[doc_id])
            pairwise_dist = pairwise_distances(bottleneck_features_train, bottleneck_features_train[doc_id].reshape(1,-1))

            indices = np.argsort(pairwise_dist.flatten())[0:num_results]
            pdists  = np.sort(pairwise_dist.flatten())[0:num_results]

            for i in range(len(indices)):
                rows = data[['medium_image_url','title','brand','color']].loc[data['asin']==asins[indices[i]]]
                for indx, row in rows.iterrows():
                    img_url_cnn.append(row['medium_image_url'])
                    title_cnn.append(row['title'])
                    color_cnn.append(row['color'])
                    brand_cnn.append(row['brand'])
                    euc_dist_cnn.append(pdists[i])

        #CNN END--------------------------------------------------------------------------------------------------------------------------





        nn = request.GET.get('items')



        
        query_img_idx = request.GET.get('scale')
        query_img_context = {

            'query_img' :data.iloc[int(query_img_idx)]['medium_image_url'],
            'query_img_title' : data.iloc[int(query_img_idx)]['title'],
            'query_img_brand' : data.iloc[int(query_img_idx)]['brand'],
            'query_img_color' : data.iloc[int(query_img_idx)]['color'],
            'query_img_price' : data.iloc[int(query_img_idx)]['formatted_price']

        }
    

        get_similar_products_cnn(int(query_img_idx), 12)
        tfidf_model(int(query_img_idx),6)
        bag_of_words_model(int(query_img_idx), 6)

        img_cnn = zip(img_url_cnn,title_cnn,color_cnn,brand_cnn,euc_dist_cnn)
        img_tfidf = zip(img_url_tfidf,title_tfidf,color_tfidf,brand_tfidf,euc_dist_tfidf)
        img_bow = zip(img_url_bow,title_bow,color_bow,brand_bow,euc_dist_bow)
        return render(request,'fashion_engine/select_img.html',{'img_bow':img_bow,'img_tfidf':img_tfidf,'img_cnn':img_cnn,'query_img_context': query_img_context})
