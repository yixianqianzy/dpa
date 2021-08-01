import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from utils import get_df, filterout, dataset_filtering, clean_string, semantic_embedding, index_encoding, semantic_embedding, dataset_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=5)

tqdm.pandas(desc="my bar!")
BASEFILE_DIR = '../../amazone_rs'  # here
PATH_word2vec = os.path.join(BASEFILE_DIR, 'google.bin')

SAVE_DIR = '../dataset/tmp_word2vec_data'
if not os.path.exists( SAVE_DIR ):
    print(f"makedir {SAVE_DIR}")
    os.makedirs(SAVE_DIR)

core = 1                # filter the dataset with x-core

# same process as 1datasetprepare
HASHPATH = {
    "ele":os.path.join(BASEFILE_DIR,"reviews_Electronics_5.json.gz"),
            "cd":os.path.join(BASEFILE_DIR,"reviews_CDs_and_Vinyl_5.json.gz"),
           "movies":os.path.join(BASEFILE_DIR,"reviews_Movies_and_TV_5.json.gz"),
           "musics":os.path.join(BASEFILE_DIR,"reviews_Digital_Music_5.json.gz"),
           "book":os.path.join(BASEFILE_DIR,"reviews_Books_5.json.gz"),
           }
all_dataset = {k:get_df(v) for k,v in HASHPATH.items()}
all_dataset_f = {k:filterout(v,13,13) for k,v in all_dataset.items()}


for dataset in all_dataset_f.keys():
    print( f"{dataset}\t"*10 )
    df_s = all_dataset_f[dataset]

    df_s['reviewText'] = df_s['reviewText'].astype(str)
    df_s['reviewText'] = df_s['reviewText'].parallel_apply( clean_string )
    df_s['reviewText'] = df_s['reviewText'].fillna('None')

    # ref: https://github.com/AkiraZC/CATN/blob/master/dataset/preprocessing.py
    vectorizer = TfidfVectorizer(max_df=0.5, stop_words={'english'}, max_features=20000)
    all_text_source = df_s['reviewText'].tolist()
    tfidf_source = vectorizer.fit_transform(all_text_source)
    vocab_dict = vectorizer.vocabulary_
    vocab_dict = set(vocab_dict.keys())
    if dataset=='book':
        print("TFIDF finished")
        df_s['reviewText'] = df_s['reviewText'].apply( lambda x:x.split() )
        print("split -> reviewText")
        df_s['reviewText'] = df_s['reviewText'].apply( lambda x:set(x) )
        print("set -> reviewText")
        df_s['reviewText'] = df_s['reviewText'].apply( lambda x:x & vocab_dict )
    else:
        print("TFIDF finished")
        df_s['reviewText'] = df_s['reviewText'].parallel_apply( lambda x:x.split() )
        print("split -> reviewText")
        df_s['reviewText'] = df_s['reviewText'].parallel_apply( lambda x:set(x) )
        print("set -> reviewText")
        df_s['reviewText'] = df_s['reviewText'].parallel_apply( lambda x:x & vocab_dict )

    print(" & vocab_dict -> reviewText")

    interaction = df_s[['reviewerID','asin','reviewText']].values.tolist()

    print('filtering data ...')
    # ipdb.set_trace()
    interaction = dataset_filtering(interaction, core)  # to remove the users and items less than core interactions
    # ipdb.set_trace()
    print('encoding data ...')
    interaction, user_review, item_review, word2num, user_id2num, item_id2num = index_encoding(interaction)
    print('loading features ...')
    # ipdb.set_trace()
    semantic_matrix = semantic_embedding(word2num, PATH_word2vec)
    train_data = dataset_sparse(interaction)

    ALL_DATA = {}
    ALL_DATA['train_data'] = train_data
    ALL_DATA['user_text'] = user_review
    ALL_DATA['item_text'] = item_review
    ALL_DATA['semantic_matrix'] = semantic_matrix.tolist()
    ALL_DATA['user_id2num'] = user_id2num
    ALL_DATA['item_id2num'] = item_id2num

    SAVE_NAME = os.path.split( HASHPATH[dataset] )[-1].replace('.json.gz', '.pkl')
    
    pickle.dump(ALL_DATA, open(os.path.join(SAVE_DIR,SAVE_NAME), 'wb'))
