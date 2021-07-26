## For TMN and TCF

import time
import os
import pandas as pd
import numpy as np
from model_TMN import *
import sys
import gc
import pickle  
import numpy as np 
import random 
import tensorflow as tf 
from tqdm import tqdm 

def print_params(para_name, para):
    for i in range(len(para)):
        print(para_name[i]+':  ',para[i])

def assignment(arr, lis):
    n = len(arr)                 
    rho = int(n / len(lis)) + 1   
    Lis = lis * rho             
    return np.array(Lis[0: n])    

def read_pkl(path):
    return pickle.load(open(path,'rb'))

def read_data(data):
    user_num = len(data)
    item_num = 0
    interactions = []
    for user in range(user_num):
        for item in data[user]:
            interactions.append((user, item))
            item_num = max(item, item_num)
    item_num += 1
    print("Successfully load")
    return(data, interactions, user_num, item_num)

model = 1           # 0:MF, 1:TMN, 2:TCF (TMF)
dataset = 1         # 0:movie, 1:video, 2:cd, 3:clothes
validate_test = 0   # 0:Validate, 1: Test
MODEL = 'TMN'

EMB_DIM = 128
BATCH_SIZE = 2048
SAMPLE_RATE = 1
N_EPOCH = 2

IF_SAVE_EMB = 1   # 1: save, otherwise: not save

DIR = '../dataset/tmp_word2vec_data/'

if __name__ == '__main__':
    ## paths of data
    SAVE_DICT = {"reviews_CDs_and_Vinyl_5.pkl":"all_cd_contentFeature.pkl",
                "reviews_Digital_Music_5.pkl":"all_musics_contentFeature.pkl",
                "reviews_Electronics_5.pkl":"all_ele_contentFeature.pkl",
                "reviews_Movies_and_TV_5.pkl":"all_movies_contentFeature.pkl",
                "reviews_Books_5.pkl":"all_book_contentFeature.pkl"}
    DATASET = list(SAVE_DICT.keys())[int(sys.argv[1])]
    print(f"{DATASET}\n"*5)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if 'Electronics' in DATASET:
        LR = 0.00005   
        LAMDA = 0.05      #
    elif 'CDs' in DATASET:
        LR = 0.00005   
        LAMDA = 0.01      #       
    else:
        LR = 0.00005   
        LAMDA = 0.01      #       


    train_path = DIR + DATASET
    SAVE_EMB_PATH =  DIR + SAVE_DICT[DATASET]

    para = [DATASET,MODEL,LR,LAMDA,EMB_DIM, BATCH_SIZE,SAMPLE_RATE,N_EPOCH]
    para_name = ['DATASET','MODEL','LR','LAMDA','EMB_DIM','BATCH_SIZE','SAMPLE_RATE','N_EPOCH',]
    ## print and save model hyperparameters
    print_params(para_name, para)
    time.sleep(10)
    if_save_emb = IF_SAVE_EMB
    ## train the model


    ## load train data
    ALL_DATA = read_pkl(train_path)
    [train_data, train_data_interaction, user_num, item_num] = read_data(ALL_DATA['train_data']) 
    
    if MODEL == 'TMN':
        text_matrix = np.array(ALL_DATA['semantic_matrix'])
        # ipdb.set_trace()

        user_review = ALL_DATA['user_text']   # 每个用户说话是不定长的
        item_review = ALL_DATA['item_text']   # 每个商品描述是不定长的


        # 只选200个词
        user_word_num = 0
        for review in user_review:
            user_word_num = max(len(review), user_word_num)
        item_word_num = 0
        for review in item_review:
            item_word_num = max(len(review), item_word_num)
        user_word_num = min(user_word_num, 200)
        item_word_num = min(item_word_num, 200)
        user_review_feature = np.ones((user_num, user_word_num))
        item_review_feature = np.ones((item_num, item_word_num))

        # ipdb.set_trace()

        for user in range(user_num):
            if len(user_review[user])!=0: user_review_feature[user] = assignment(user_review_feature[user], user_review[user])
            else: user_review_feature[user] = assignment(user_review_feature[user], [0])
        for item in range(item_num):
            if len(item_review[item])!=0: item_review_feature[item] = assignment(item_review_feature[item], item_review[item])
            else: item_review_feature[item] = assignment(item_review_feature[item], [0])
    
    print("Strat training")
    ## define the model
    if MODEL == 'TMN': model = model_TMN(n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, text_embeddings = text_matrix, user_word_num = user_word_num, item_word_num = item_word_num)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    ## split the training samples into batches
    batches = list(range(0, len(train_data_interaction), BATCH_SIZE))
    batches.append(len(train_data_interaction))

    ## training iteratively
    for epoch in range(N_EPOCH):
        ITERATION = len(batches)
        with tqdm(total = ITERATION, desc = f'Epoch{epoch}' ) as pbar:
            for batch_num in range(ITERATION-1):
                # Description will be displayed on the left

                train_batch_data = []
                if MODEL == 'TMN':
                    user_review_batch = np.ones(((1+SAMPLE_RATE)*(batches[batch_num+1]-batches[batch_num]), user_word_num))
                    item_review_batch = np.ones(((1+SAMPLE_RATE)*(batches[batch_num+1]-batches[batch_num]), item_word_num))
                num = 0
                for sample in range(batches[batch_num], batches[batch_num+1]):
                    user, pos_item = train_data_interaction[sample]
                    sample_num = 0
                    train_batch_data.append([user, pos_item, 1]) 
                    if MODEL == 'TMN':
                        user_review_batch[num] = user_review_feature[user]
                        item_review_batch[num] = item_review_feature[pos_item]
                    num += 1
                    while sample_num < SAMPLE_RATE:
                        neg_item = int(random.uniform(0, item_num))
                        if not (neg_item in train_data[user]):
                            sample_num += 1
                            train_batch_data.append([user, neg_item, 0])
                            if MODEL == 'TMN'
                                user_review_batch[num] = user_review_feature[user]
                                item_review_batch[num] = item_review_feature[neg_item]
                            num += 1
                train_batch_data = np.array(train_batch_data)
                try:
                    _, loss = sess.run([model.updates, model.loss],
                                    feed_dict={model.users: train_batch_data[:, 0],
                                                model.items: train_batch_data[:, 1],
                                                model.label: train_batch_data[:, 2],
                                                model.user_word: user_review_batch,
                                                model.item_word: item_review_batch})
                except:
                    _, loss = sess.run([model.updates, model.loss],
                                    feed_dict={model.users: train_batch_data[:, 0],
                                                model.items: train_batch_data[:, 1],
                                                model.label: train_batch_data[:, 2]})
                # Postfix will be displayed on the right,
                # formatted automatically based on argument's datatype
                pbar.set_postfix( {'loss' : '{0:1.5f}'.format(loss)} )
                pbar.update(1)


    if if_save_emb == 1:
        try:
            user_text_embedding = np.zeros((user_num, np.shape(text_matrix)[1]))
            item_text_embedding = np.zeros((item_num, np.shape(text_matrix)[1]))
            user_batch_list = list(range(0, user_num, 500))
            user_batch_list.append(user_num)
            item_batch_list = list(range(0, item_num, 500))
            item_batch_list.append(item_num)
            for u in range(len(user_batch_list) - 1):
                u1, u2 = user_batch_list[u], user_batch_list[u + 1]
                user_batch = np.array(range(u1, u2))
                user_review_batch = user_review_feature[u1: u2]
                u_text_embedding = sess.run([model.u_text_embeddings],
                                                feed_dict={model.users: user_batch,
                                                        model.user_word: user_review_batch})
                user_text_embedding[u1: u2] = u_text_embedding[0]
            for i in range(len(item_batch_list) - 1):
                i1, i2 = item_batch_list[i], item_batch_list[i + 1]
                item_batch = np.array(range(i1, i2))
                item_review_batch = item_review_feature[i1: i2]
                i_text_embedding = sess.run([model.i_text_embeddings],
                                                feed_dict={model.items: item_batch,
                                                        model.item_word: item_review_batch})
                item_text_embedding[i1: i2] = i_text_embedding[0]
        except:
            user_embedding, item_embedding = sess.run([model.user_embeddings, model.item_embeddings])     

        if MODEL == 'TMN':
            user_id2num = ALL_DATA['user_id2num']
            item_id2num = ALL_DATA['item_id2num']

            save_user_text_embedding = {k:user_text_embedding[v] for k,v in user_id2num.items()}
            save_item_text_embedding = {k:item_text_embedding[v] for k,v in item_id2num.items()}

            pickle.dump({'user':save_user_text_embedding, 'item':save_item_text_embedding}, open(SAVE_EMB_PATH, 'wb'))

        # if MODEL == 'TMF':
        #     save_embeddings([user_embedding.tolist(), item_embedding.tolist()], save_latant_embeddings_path)
        try: del u_text_embedding, i_text_embedding, user_text_embedding, item_text_embedding
        except: del user_embedding, item_embedding
    del model, loss, _, sess
    gc.collect()
    