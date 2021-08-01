import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import pandas as pd
import json
from tqdm import tqdm
import logging
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=10)

import ipdb 


def get_support_query_index(dataset_split_y):
    """
    len(support):len(query) = 1:4
    len(support & +):len(query & +) = 1:4
    Args:
    dataset_split_y: label
    """
    all_length = len(dataset_split_y)
    support_length = round(all_length*0.25)
    query_length = all_length - support_length
    true_index = list(np.argwhere(dataset_split_y==1).flatten())
    false_index = list(np.argwhere(dataset_split_y==0).flatten())
    random.shuffle(true_index),random.shuffle(false_index)    
    support_true_split_loc = round(len(true_index)*0.25)  # 正样本的分割位置
    support_false_split_loc = support_length-support_true_split_loc   #负样本的分割位置

    support_true_split_loc = support_true_split_loc if support_true_split_loc!=0 else 1

    support_index = true_index[:support_true_split_loc]   + false_index[:support_false_split_loc]
    query_index   = true_index[support_true_split_loc:] + false_index[support_false_split_loc:]

    random.shuffle(support_index),random.shuffle(query_index)
    return support_index, query_index

def get_feature_dict(args, name="", datapath=None, bathpath = '../dataset/tmp_word2vec_data'):
    if datapath==None:
        datapath = os.path.join(bathpath, 'all_'+ args.dataset +args.featuremarker)

    feature_dict = pickle.load(open(datapath, "rb")) 
    user_feature_dict = feature_dict['user']
    item_feature_dict = feature_dict['item']
    if name!="":
        user_feature_dict = {(name+'_'+k):v for k,v in user_feature_dict.items()}
        item_feature_dict = {(name+'_'+k):v for k,v in item_feature_dict.items()}

    logging.info("successfully loading datapath {}".format(datapath))
    return user_feature_dict, item_feature_dict


def get_dict_domain_rating(path, name=""):
    """
    given csv file path, return two dictionary,
    1. dict[user_name] = click item
    2. dict[user_name] = corresponding rate
    """
    dict_domain_rating = {}
    dict_domain_rating_y = {}
    domain_rating = pd.read_csv(path)    # book_base_ele_ratings.csv
    if len(domain_rating.columns) == 4:
        logging.info("ATTENTION: {} ,the first index is set to index".format(domain_rating.columns))
        domain_rating = pd.read_csv(path, index_col=0)    # book_base_ele_ratings.csv
    if list(domain_rating.columns) != ['reviewerID', 'itemID', 'ratings']:
        domain_rating.columns = ['reviewerID', 'itemID', 'ratings']
    """normalize into [0, 1] from [0, max_rating]"""
    domain_rating.loc[domain_rating['ratings']!=0,'ratings']=1
    logging.info( "ATTENTION: file {}, max rate: {}, min rate: {}".format( path, max(domain_rating['ratings']), min(domain_rating['ratings']) ) )
    if name!="":
        domain_rating['reviewerID'] = name + '_' + domain_rating['reviewerID']
    for dfGroupBy in domain_rating.groupby(['reviewerID']):  # 每一个用户
        now_user_name = dfGroupBy[0]
        now_df = dfGroupBy[1]
        dict_domain_rating[now_user_name] = []
        dict_domain_rating_y[now_user_name] = []
        for _, row in now_df.iterrows():
            dict_domain_rating[now_user_name].append(row['itemID'])
            dict_domain_rating_y[now_user_name].append(row['ratings'])
    return dict_domain_rating, dict_domain_rating_y

def get_dict_eval_rating(path, name=""):
    """
    测试部分有不同的生成方法
    dict[USERID]={support:[[+,-,...,-], [+,-,...,-], [+,-,...,-]],
                  query:  [[+,-,...,-], [+,-,...,-], [+,-,...,-]] }
    """
    dict_domain_rating = {}
    domain_rating = json.load(open(path, "rb"))

    logging.info( "ATTENTION: domain {} data -> total user: {}".format( name, len(domain_rating) ) )
    if name!="":
        domain_rating = {name+k:v for k,v in domain_rating.items()}
    return domain_rating


def read_self_pkl(path, name="", n_support_size=5):
    dataset_split = {}
    dataset_split_y = {}
    # tmp_now_rw = pickle.load(open(path, "rb"))  
    tmp_now_rw = np.load(path, allow_pickle=True).item()

    if name!="":
        now_rw = {(name+"_"+k):v for k,v in tmp_now_rw.items()}
    else:
        now_rw = tmp_now_rw

    for k,v in now_rw.items():
        dataset_split[k]=np.array(v['support']).flatten()
        dataset_split_y[k]=np.array(len(v['support']) * ([1] + [0]*n_support_size))

    return dataset_split, dataset_split_y


class Train_Dataset(Dataset):
    def __init__(self, args, test_way=None):
        super(Train_Dataset, self).__init__()
        self.args = args

        self.user_feature_dict = {}
        self.item_feature_dict = {}

        assert test_way is None 
        self.state = 'Rw'

#################################################################################################################################
#################################################################################################################################
# target domain 

        now_domain = 'target'
        logging.info("Loading dict")

        now_domain_user_feature_dict, now_domain_item_feature_dict = get_feature_dict(args, name="")


        self.user_feature_dict = now_domain_user_feature_dict
        self.item_feature_dict = now_domain_item_feature_dict

        logging.info("check now_domain_user_feature_dict - {} feature range: {}".format( now_domain, \
            [ np.max( list(now_domain_user_feature_dict[list(now_domain_user_feature_dict.keys())[x]]) ) for x in range(20)]) )
        logging.info("check now_domain_item_feature_dict - {} feature range: {}".format( now_domain, \
            [ np.max( list(now_domain_item_feature_dict[list(now_domain_item_feature_dict.keys())[x]]) ) for x in range(20)]) )
        
        logging.info("Finish loading dict") 
########################################################
        self.dataset_split = {}
        self.dataset_split_y = {}
        
        path_domain_rating = os.path.join(args.datadir, f'{args.dataset}_test_data',f'{self.state}_valid.npy')
        logging.info('Preparing data {}'.format(path_domain_rating))
        dict_domain_rating, dict_domain_rating_y = read_self_pkl(path_domain_rating, n_support_size=5)
        logging.info("{} has {} data".format(now_domain, len(dict_domain_rating)))

        self.dataset_split = dict(self.dataset_split, **dict_domain_rating)
        self.dataset_split_y = dict(self.dataset_split_y, **dict_domain_rating_y)

        assert len(self.dataset_split) == len(self.dataset_split_y)
        logging.info("update dataset to {} users".format(len(self.dataset_split)))

#################################################################################################################################
#################################################################################################################################
# source domain
        # 读取四个数据集
        logging.info("{}{}".format("="*50, self.args.supportdata))
        if (self.args.supportdata == "None") or (test_way is not None):
            self.args.supportdata = []
        else:
            self.args.supportdata = self.args.supportdata.split('+')

        for now_domain in tqdm(self.args.supportdata):   # ele+movies+musics
            
            # logging.info("Loading dict")
            source_name = now_domain
            target_name = args.dataset
            basedir = os.path.join(args.datadir, f'{now_domain}_{args.dataset}')

            Path_domain_rating = os.path.join( args.g_path, source_name+'_'+target_name + args.g_name + '_generation_rw.csv')

            logging.info("!! Buliding data from {}".format(Path_domain_rating))
            dict_domain_rating, dict_domain_rating_y = get_dict_domain_rating(Path_domain_rating, name=now_domain)

            logging.info("{} has {} data".format(now_domain, len(dict_domain_rating)))

            self.dataset_split = dict(self.dataset_split, **dict_domain_rating)
            self.dataset_split_y = dict(self.dataset_split_y, **dict_domain_rating_y)

            assert len(self.dataset_split) == len(self.dataset_split_y)
            logging.info("update dataset to {} users".format(len(self.dataset_split)))
#################################################################################################################################
#################################################################################################################################

        self.final_index = []

        pool_alllength = [ len(v) for k,v in self.dataset_split.items()]
        min_thre = int(np.quantile(pool_alllength,0.1))
        max_thre = int(np.quantile(pool_alllength,0.9))

        for _, user_id in enumerate(tqdm(list(self.dataset_split.keys()))):   # 遍历所有的用户
            seen_movie_len = len(self.dataset_split[str(user_id)])

            if seen_movie_len < min_thre or seen_movie_len > max_thre:
                continue
            else:
                self.final_index.append(user_id)          
        logging.info("low bound is {}, high bound is {}".format(min_thre, max_thre))                    
        logging.info("select {} users from {} users".format(len(self.final_index), len(self.dataset_split)))
        
        
    def __getitem__(self, item):
        u_name = self.final_index[item]  # user_id
        short_u_name = self.final_index[item].split('_')[-1]  # user_id
        tmp_x = np.array(self.dataset_split[u_name])      # 当前user name所有点击商品
        tmp_y = np.array(self.dataset_split_y[u_name])    # 当前user name所有点击商品 score

        now_domain = u_name.split('_')[0]

        now_domain_user_feature_dict = self.user_feature_dict
        now_domain_item_feature_dict = self.item_feature_dict

        support_index, query_index = get_support_query_index(tmp_y)
        # print(tmp_y[support_index].shape,tmp_y[query_index].shape)
        # print( np.where(tmp_y[support_index]==1)[0].shape[0], np.where(tmp_y[query_index]==1)[0].shape[0])

        support_x_app = None
        # for m_id in tmp_x[indices[:-10]]:  # 这边的数字要设置一下
        for m_id in tmp_x[support_index]:  # 这边的数字要设置一下
            # m_id = now_domain + '_' + m_id
            # 读取feature
            # print(u_name, m_id)
            tmp_x_converted = torch.cat( ( torch.Tensor(now_domain_item_feature_dict[m_id]).unsqueeze(0), 
                                            torch.Tensor(now_domain_user_feature_dict[short_u_name]).unsqueeze(0)), 1)
            try:
                support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
            except:
                support_x_app = tmp_x_converted
                
        query_x_app = None
        for m_id in tmp_x[query_index]:
            # m_id = now_domain + '_' + m_id
            tmp_x_converted = torch.cat( ( torch.Tensor(now_domain_item_feature_dict[m_id]).unsqueeze(0), 
                                            torch.Tensor(now_domain_user_feature_dict[short_u_name]).unsqueeze(0)), 1)
            try:
                query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
            except:
                query_x_app = tmp_x_converted
        support_y_app = torch.FloatTensor(tmp_y[support_index])
        query_y_app = torch.FloatTensor(tmp_y[query_index])

        return support_x_app, support_y_app.view(-1,1), query_x_app, query_y_app.view(-1,1), now_domain
        
    def __len__(self):   # 用户数量
        
        return len(self.final_index)





#### abla_true_data

class Train_Dataset_abla(Dataset):
    def __init__(self, args, test_way=None):
        super(Train_Dataset_abla, self).__init__()
        self.args = args

        self.user_feature_dict = {}
        self.item_feature_dict = {}

        assert test_way is None 
        assert args.abla_true_data == True
        self.state = 'Rw'
# target domain 

        now_domain = 'target'
        logging.info("Loading dict")

        now_domain_user_feature_dict, now_domain_item_feature_dict = get_feature_dict(args)

        self.user_feature_dict[now_domain] = now_domain_user_feature_dict
        self.item_feature_dict[now_domain] = now_domain_item_feature_dict

        logging.info("check now_domain_user_feature_dict - {} feature range: {}".format( now_domain, \
            [ np.max( list(now_domain_user_feature_dict[list(now_domain_user_feature_dict.keys())[x]]) ) for x in range(20)]) )
        logging.info("check now_domain_item_feature_dict - {} feature range: {}".format( now_domain, \
            [ np.max( list(now_domain_item_feature_dict[list(now_domain_item_feature_dict.keys())[x]]) ) for x in range(20)]) )
        
        logging.info("Finish loading dict") 
        ########################################################
        self.dataset_split = {}
        self.dataset_split_y = {}
        
        path_domain_rating = os.path.join(args.datadir, f'{args.dataset}_test_data',f'{self.state}_valid.npy')
        logging.info('Preparing data {}'.format(path_domain_rating))
        dict_domain_rating, dict_domain_rating_y = read_self_pkl(path_domain_rating, name=now_domain, n_support_size=5)
        logging.info("{} has {} data".format(now_domain, len(dict_domain_rating)))

        self.dataset_split = dict(self.dataset_split, **dict_domain_rating)
        self.dataset_split_y = dict(self.dataset_split_y, **dict_domain_rating_y)

        assert len(self.dataset_split) == len(self.dataset_split_y)
        logging.info("update dataset to {} users".format(len(self.dataset_split)))

# source domain
        # 读取四个数据集
        logging.info("{}{}".format("="*50, self.args.supportdata))
        if (self.args.supportdata == "None") or (test_way is not None):
            self.args.supportdata = []
        else:
            self.args.supportdata = self.args.supportdata.split('+')

        for now_domain in tqdm(self.args.supportdata):   # ele+movies+musics
            
            # logging.info("Loading dict")
            source_name = now_domain
            target_name = args.dataset
            basedir = os.path.join(args.datadir, f'S_{now_domain}_T_{args.dataset}')

            Path_domain_rating = os.path.join( basedir, source_name+'_based_'+target_name+'_ratings.csv')     
            logging.info("!! Buliding data from {}".format(Path_domain_rating))
            dict_domain_rating, dict_domain_rating_y = get_dict_domain_rating(Path_domain_rating, name=now_domain)

            logging.info("{} has {} data".format(now_domain, len(dict_domain_rating)))

            self.dataset_split = dict(self.dataset_split, **dict_domain_rating)
            self.dataset_split_y = dict(self.dataset_split_y, **dict_domain_rating_y)

            assert len(self.dataset_split) == len(self.dataset_split_y)
            logging.info("update dataset to {} users".format(len(self.dataset_split)))

            feature_dict_path = os.path.join('../dataset/tmp_word2vec_data', 'all_'+ source_name +args.featuremarker)
            now_domain_user_feature_dict, now_domain_item_feature_dict = get_feature_dict(args, name="", datapath=feature_dict_path)

            self.user_feature_dict[now_domain] = now_domain_user_feature_dict
            self.item_feature_dict[now_domain] = now_domain_item_feature_dict


        self.final_index = []

        pool_alllength = [ len(v) for k,v in self.dataset_split.items()]
        min_thre = int(np.quantile(pool_alllength,0.1))
        max_thre = int(np.quantile(pool_alllength,0.9))


        for _, user_id in enumerate(tqdm(list(self.dataset_split.keys()))):   # 遍历所有的用户
            seen_movie_len = len(self.dataset_split[str(user_id)])

            if seen_movie_len < min_thre or seen_movie_len > max_thre:
                continue
            else:
                self.final_index.append(user_id)          
        logging.info("low bound is {}, high bound is {}".format(min_thre, max_thre))                    
        logging.info("select {} users from {} users".format(len(self.final_index), len(self.dataset_split)))
        
        
    def __getitem__(self, item):
        u_name = self.final_index[item]  # user_id
        short_u_name = self.final_index[item].split('_')[-1]  # user_id
        tmp_x = np.array(self.dataset_split[u_name])      # 当前user name所有点击商品
        tmp_y = np.array(self.dataset_split_y[u_name])    # 当前user name所有点击商品 score

        now_domain = u_name.split('_')[0]

        now_domain_user_feature_dict = self.user_feature_dict[now_domain]
        now_domain_item_feature_dict = self.item_feature_dict[now_domain]

        support_index, query_index = get_support_query_index(tmp_y)

        support_x_app = None

        # print(u_name, short_u_name, tmp_x[support_index], list(now_domain_user_feature_dict.keys())[:3])

        for m_id in tmp_x[support_index]:  
            # m_id = now_domain + '_' + m_id
            tmp_x_converted = torch.cat( ( torch.Tensor(now_domain_item_feature_dict[m_id]).unsqueeze(0), 
                                            torch.Tensor(now_domain_user_feature_dict[short_u_name]).unsqueeze(0)), 1)
            try:
                support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
            except:
                support_x_app = tmp_x_converted
                
        query_x_app = None
        for m_id in tmp_x[query_index]:
            # m_id = now_domain + '_' + m_id
            tmp_x_converted = torch.cat( ( torch.Tensor(now_domain_item_feature_dict[m_id]).unsqueeze(0), 
                                            torch.Tensor(now_domain_user_feature_dict[short_u_name]).unsqueeze(0)), 1)
            try:
                query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
            except:
                query_x_app = tmp_x_converted
        support_y_app = torch.FloatTensor(tmp_y[support_index])
        query_y_app = torch.FloatTensor(tmp_y[query_index])

        return support_x_app, support_y_app.view(-1,1), query_x_app, query_y_app.view(-1,1), now_domain
        
    def __len__(self):   # 用户数量
        
        return len(self.final_index)




class Test_Dataset(Dataset):
    def __init__(self, args, partition='valid', test_way=None):
        super(Test_Dataset, self).__init__()
        self.args = args

        self.user_feature_dict = {}
        self.item_feature_dict = {}

        assert test_way is not None
        if test_way == 'old':
            self.state = 'Rw'        
        elif test_way == 'new_user':
            self.state = 'Rcu'
        elif test_way == 'new_item':
            self.state = 'Rci'
        elif test_way == 'new_item_user':
            self.state = 'Rcui'
        else:
            exit() 

        self.now_domain = 'target'
        logging.info("Loading dict")    

        now_domain_user_feature_dict, now_domain_item_feature_dict = get_feature_dict(args, name="")

        self.user_feature_dict[self.now_domain] = now_domain_user_feature_dict
        self.item_feature_dict[self.now_domain] = now_domain_item_feature_dict
        
        logging.info("Finish loading dict") 
########################################################

        path_domain_rating = os.path.join(args.datadir, f'{args.dataset}_test_data',f'{self.state}_{partition}.npy')
        logging.info('Preparing data {}'.format(path_domain_rating))

        self.dict_domain_rating = np.load(path_domain_rating, allow_pickle=True).item() 

        logging.info("{} has {} data".format(self.now_domain, len(self.dict_domain_rating)))


        self.n_support_size = len(self.dict_domain_rating[list(self.dict_domain_rating.keys())[0]]['support'][0])-1 # 负样本大小
        self.n_query_size   = len(self.dict_domain_rating[list(self.dict_domain_rating.keys())[0]]['query'])-1 # 负样本大小

        self.final_index = list(self.dict_domain_rating.keys())

        logging.info("select {} data, negative sample size of support: {}, negative sample size of query: {}".format(
            len(self.final_index), self.n_support_size, self.n_query_size))

    def __getitem__(self, item):
        u_name = self.final_index[item]  # user_id
        # print("u_name", u_name)
        tmp_support = self.dict_domain_rating[u_name]['support']
        tmp_query = self.dict_domain_rating[u_name]['query']
        
        if len(tmp_support)>5:
            tmp_support = tmp_support[:5]   # same as maml

        tmp_x_support = np.array(tmp_support).flatten()      # 当前user name所有点击商品
        tmp_y_support = np.array(len(tmp_support) * ([1] + [0]*self.n_support_size))

        tmp_x_query = np.array(tmp_query).flatten()      # 当前user name所有点击商品
        tmp_y_query = np.array([1] + [0]*self.n_query_size) 

        now_domain = self.now_domain
        now_domain_user_feature_dict = self.user_feature_dict[now_domain]
        now_domain_item_feature_dict = self.item_feature_dict[now_domain]

        support_x_app = None
        # for m_id in tmp_x[indices[:-10]]:  # 这边的数字要设置一下
        for m_id in tmp_x_support:  # 这边的数字要设置一下
            # m_id = now_domain + '_' + m_id
            # 读取feature
            tmp_x_converted = torch.cat( ( torch.Tensor(now_domain_item_feature_dict[m_id]).unsqueeze(0), 
                                            torch.Tensor(now_domain_user_feature_dict[u_name]).unsqueeze(0)), 1)
            try:
                support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
            except:
                support_x_app = tmp_x_converted
                
        query_x_app = None
        for m_id in tmp_x_query:
            # m_id = now_domain + '_' + m_id
            tmp_x_converted = torch.cat( ( torch.Tensor(now_domain_item_feature_dict[m_id]).unsqueeze(0), 
                                            torch.Tensor(now_domain_user_feature_dict[u_name]).unsqueeze(0)), 1)
            try:
                query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
            except:
                query_x_app = tmp_x_converted
                
        support_y_app = torch.FloatTensor(tmp_y_support)
        query_y_app = torch.FloatTensor(tmp_y_query)
        
        return support_x_app, support_y_app.view(-1,1), query_x_app, query_y_app.view(-1,1), now_domain
        
    def __len__(self):   # 用户数量
        
        return len(self.final_index)
