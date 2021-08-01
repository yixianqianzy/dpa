import os
import pickle
import logging
import pandas as pd 
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=5)
import numpy as np 

class Dataset(object):
    def __init__(self, args):

        self.args = args 

        self.source_rating, self.source_item_feature, self.source_user_feature,   \
        self.target_rating, self.target_item_feature, self.target_user_feature,   \
        self.source_item2id, self.target_item2id, self.user2id= self.load_data()

        self.source_set = self.source_item2id.keys()
        self.target_set = self.target_item2id.keys()
        self.num_user = len(set(self.source_rating['reviewerID']))
        self.num_source_item = len(self.source_set)
        self.num_target_item = len(self.target_set)

        logging.info("user number:{}, source item number:{}, target item number:{}".format(self.num_user, self.num_source_item, self.num_target_item))
        
        # dataframe编码成字典，方便sparse matrix的生成f
        logging.info('dict generating')
        ## source_rating
        self.dict_source_rating = {}
        for index, row in self.source_rating.iterrows():
            now_user_id = self.user2id[row['reviewerID']]
            now_item_id = self.source_item2id[row['itemID']]
            if now_user_id not in self.dict_source_rating:
                self.dict_source_rating[now_user_id]={}
            self.dict_source_rating[now_user_id][now_item_id]=row['ratings']    
        # print(self.dict_source_rating.keys())
        ## target_rating
        self.dict_target_rating = {}
        for index, row in self.target_rating.iterrows():
            now_user_id = self.user2id[row['reviewerID']]
            now_item_id = self.target_item2id[row['itemID']]
            if now_user_id not in self.dict_target_rating:
                self.dict_target_rating[now_user_id]={}
            self.dict_target_rating[now_user_id][now_item_id]=row['ratings'] 
        # print(self.dict_target_rating.keys())       
        logging.info('finished dict generating')


    def load_data(self):
        logging.info('loading data...')
        save_path = os.path.join(os.getcwd(), 'processed_data', self.args.dataset+'.npy')
        if not os.path.exists(save_path) or self.args.reset_dataset:
            try:
                os.makedirs(os.path.join(os.getcwd(), 'processed_data'))
            except:
                pass 
    
            thre = 1

            source_name = self.args.dataset.split('_')[0]
            target_name = self.args.dataset.split('_')[1]
            
            Path_source_rating = os.path.join(self.args.datadir, f"S_{source_name}_T_{target_name}", source_name+'_based_'+target_name+'_ratings.csv') 

            Path_source_feature = os.path.join("../dataset/tmp_word2vec_data", 'all_' + source_name + '_contentFeature.pkl')

            Path_target_rating = os.path.join(self.args.datadir, f"S_{source_name}_T_{target_name}", target_name+'_based_'+source_name+'_ratings.csv')  

            Path_target_feature = os.path.join("../dataset/tmp_word2vec_data", 'all_' + target_name + '_contentFeature.pkl')

            source_rating = pd.read_csv(Path_source_rating)
            if list(source_rating.columns) == ['reviewerID','asin','overall']: source_rating.columns=['reviewerID','itemID','ratings']


            source_rating.loc[source_rating['ratings']>=thre, 'ratings'] = 1   # transform
            source_rating = source_rating[source_rating['ratings'] == 1]    # filter

            source_feature = pickle.load(open(Path_source_feature, 'rb'))
            source_item_feature = source_feature['item']
            source_user_feature = source_feature['user']
            
            target_rating = pd.read_csv(Path_target_rating)
            if list(target_rating.columns) == ['reviewerID','asin','overall']: target_rating.columns=['reviewerID','itemID','ratings']

            target_rating.loc[target_rating['ratings']>=thre, 'ratings'] = 1
            target_rating = target_rating[target_rating['ratings'] == 1]

            target_feature = pickle.load(open(Path_target_feature, 'rb'))
            target_item_feature = target_feature['item']
            target_user_feature = target_feature['user']


            # encoder
            source_item2id = dict()
            target_item2id = dict()
            user2id = dict()
            for idx, now_user in enumerate(set(source_rating['reviewerID'])):
                user2id[now_user] = idx
            
            for idx, now_item in enumerate(set(source_rating['itemID'])):
                source_item2id[now_item] = idx
            
            # for idx, now_item in enumerate(set(target_rating['itemID'])):
            #     target_item2id[now_item] = idx

            target_item2id = pickle.load(open('../dataset/all_target_domain_item_dict.pkl', 'rb'))[target_name]

            # ipdb.set_trace()

            logging.info("processing features")
            # feature csv 转换成id
            set_source_rating_item = source_rating['itemID'].tolist()
            set_source_rating_user = source_rating['reviewerID'].tolist()
            set_target_rating_item = target_rating['itemID'].tolist()
            set_target_rating_user = target_rating['reviewerID'].tolist()

            source_item_feature = {source_item2id[k]:v for k,v in source_item_feature.items() if k in set_source_rating_item}

            source_user_feature = {user2id[k]:v for k,v in source_user_feature.items() if k in set_source_rating_user}

            target_item_feature = {target_item2id[k]:v for k,v in target_item_feature.items() if k in set_target_rating_item}

            target_user_feature = {user2id[k]:v for k,v in target_user_feature.items() if k in set_target_rating_user}

            logging.info("finished processing features")

            ALL_DATA = [source_rating, source_item_feature, source_user_feature, \
                target_rating, target_item_feature, target_user_feature, \
                    source_item2id, target_item2id, user2id]

            np.save(save_path, ALL_DATA)
        else:
            logging.info("directly load {}".format(save_path))
            source_rating, source_item_feature, source_user_feature, \
                target_rating, target_item_feature, target_user_feature, \
                    source_item2id, target_item2id, user2id = np.load(save_path, allow_pickle=True)

        logging.info('finished loading data...')

        return source_rating, source_item_feature, source_user_feature, \
                target_rating, target_item_feature, target_user_feature, \
                    source_item2id, target_item2id, user2id

    def get_train_indices(self, domain):
        row, col, data = [], [], []
        assert domain in ['source', 'target']
        rating_dict = eval(f'self.dict_{domain}_rating') 
        for now_user in range(self.num_user):
            for now_item in rating_dict[now_user]:
                row.append(now_user)
                col.append(now_item)
                data.append(rating_dict[now_user][now_item])
        return data, row, col 
    
