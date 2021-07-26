
# coding=UTF-8
import os
import gzip
import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split


BASEFILE_DIR = "../../amazone_rs"   # change here
SAVE_DIR = "../dataset/"

def get_df(path_s):
    def parse(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)

    def get_raw_df(path):
        df = {}
        for i, d in tqdm(enumerate(parse(path)), ascii=True):
            if "" not in d.values():
                df[i] = d
        df = pd.DataFrame.from_dict(df, orient='index')
        # df = df[["reviewerID", "asin", "reviewText", "overall"]]
        return df

    csv_path_s = path_s.replace('.json.gz', '.csv')

    if os.path.exists(csv_path_s):
        df_s = pd.read_csv(csv_path_s)
        print('Load raw data from %s.' % csv_path_s)
    else:
        df_s = get_raw_df(path_s)

        df_s.to_csv(csv_path_s, index=False)
        print('Build raw data to %s.' % csv_path_s)
    
    return df_s

def filterout(df, min_u=5, min_i=5):
    # init 
    if (len(df.columns)==3) and ('itemID' in df.columns):
        pass
    else:
        if 'asin' in df.columns: df = df[['reviewerID','asin','overall']]
        if list(df.columns)==['reviewerID','asin','overall']: df.columns=['reviewerID','itemID','ratings']
    before = len(df)
    
    index = df[["ratings", "itemID"]].groupby('itemID').count() >= min_u
    item = set(index[index['ratings'] == True].index)
    
    index = df[["ratings", "reviewerID"]].groupby('reviewerID').count() >= min_i
    user = set(index[index['ratings'] == True].index)
    
    
    df = df[df['itemID'].isin(item)]
    df = df[df['reviewerID'].isin(user)]
    
    after = len(df)
    print(f"before:{before}-after:{after} = {after/before}")
    return df


def get_convertdf(df, min_v = 13, max_v = 100):
    before = len(df['reviewerID'].drop_duplicates())
    count_u_df = df.groupby('reviewerID')['itemID'].count()
    filter_df = (count_u_df>=min_v) & (count_u_df<=max_v)
    save_u = filter_df[filter_df==True].index
    df = df[df['reviewerID'].isin( list(save_u) )]
    after = len(df['reviewerID'].drop_duplicates())
    
    print(f"User number: before:{before}-after:{after} = {after/before}")
    
    return df


def trans_standart_test(df, save_name, path):
    num_user = df.groupby(['reviewerID']).size()
    item_name = df.groupby(['itemID']).size().index.values
    user_name = num_user.index.values
    df_support = []
    df_query = []
    for i in tqdm(range(len(user_name))):
        df_user_all = df[(df['reviewerID'].isin([user_name[i]]))].values.tolist()
        df_item_all = df[(df['reviewerID'].isin([user_name[i]]))].itemID.tolist()
        num_negtive = 5 * num_user[i]
        negetive_item = list(set(item_name) - set(df_item_all))
        negetive_item = negetive_item[:num_negtive] 

        #positive sample   
        for j in range(0,10):
            df_query.append(df_user_all[j])
        for j in range(10,num_user[i]):
            df_support.append(df_user_all[j])
        #negetive sample
        
        for j in range(0,num_user[i]):
            df_query.append([user_name[i],negetive_item[j],0.0])
        for j in range(num_user[i],num_negtive):
            df_support.append([user_name[i],negetive_item[j],0.0])

    df_support = pd.DataFrame(df_support)
    df_query = pd.DataFrame(df_query)
    df_support = df_support.sample(frac=1.0)
    df_query = df_query.sample(frac=1.0)
    df_support.columns=['reviewerID','itemID','ratings']
    df_query.columns=['reviewerID','itemID','ratings']

    df = df_query.append(df_support)

    df.loc[df["ratings"]!=0,'ratings']=1
    print(f"\t --> {len(df[df['ratings']==1])/len(df[df['ratings']==0])}")

    all_item_name = set(df["itemID"])
    all_dict={}
    """
    {
    user_id1: [[+,-,-,-,-,-],[+,-,-,-,-,-],[+,-,-,-,-,-]],
    user_id2: [[+,-,-,-,-,-],[+,-,-,-,-,-],[+,-,-,-,-,-]]
    }
    """
    all_item = set(df["itemID"])
    drop_user_num = 0
    for name, now_group in tqdm(df.groupby("reviewerID") ):
        pos_item = now_group.loc[now_group['ratings']==1,"itemID"].tolist()
        if len(pos_item)>=2:
            all_dict[name] = {"support":[],"query":[]}

            neg_item = now_group.loc[now_group['ratings']==0,"itemID"].tolist()
            random.shuffle(pos_item), random.shuffle(neg_item)

            for idx, now_p in enumerate(pos_item):
                if idx==0:
                    all_dict[name]["query"] = [now_p] + random.sample( list(all_item - set(pos_item)), 99)
                else:
                    mini_group = [now_p] + neg_item[(5*idx):(5*idx+5)]
                    all_dict[name]["support"].append(mini_group)
        else:
            # drop this user
            drop_user_num = drop_user_num+1
    print(f"\t drop user {drop_user_num}/{len(all_dict)} = {drop_user_num/len(all_dict)}")
    
    
    valid_save_path = os.path.join(path, f"{save_name}_valid.npy")
    test_save_path  = os.path.join(path, f"{save_name}_test.npy")
    
    all_user_name = list(all_dict.keys())
    random.shuffle(all_user_name)
    
    valid_user = all_user_name[: int( len(all_user_name)/2 )]
    test_user = [x for x in all_user_name if x not in valid_user]
    
    valid_dict = {k:v for k,v in all_dict.items() if k in valid_user}
    test_dict = {k:v for k,v in all_dict.items() if k in test_user}
    
    with open(valid_save_path, "wb") as f:
        np.save(f, arr=valid_dict, allow_pickle=True)
    with open(test_save_path, "wb") as f:
        np.save(f, arr=test_dict, allow_pickle=True)

    return all_dict


HASHPATH = {
    "ele":os.path.join(BASEFILE_DIR,"reviews_Electronics_5.json.gz"),
            "cd":os.path.join(BASEFILE_DIR,"reviews_CDs_and_Vinyl_5.json.gz"),
           "movies":os.path.join(BASEFILE_DIR,"reviews_Movies_and_TV_5.json.gz"),
           "musics":os.path.join(BASEFILE_DIR,"reviews_Digital_Music_5.json.gz"),
           "book":os.path.join(BASEFILE_DIR,"reviews_Books_5.json.gz"),
           }



all_dataset = {k:get_df(v) for k,v in HASHPATH.items()}


all_dataset_f = {k:filterout(v,13,13) for k,v in all_dataset.items()}


target_domain = ["cd"]
source_domain = ['ele', 'musics', 'movies']
all_ts_combination = [[t,s] for s in source_domain for t in target_domain]
all_ts_combination = all_ts_combination
print(f"all_ts_combination: {all_ts_combination}")


POOL_UNSHARE = []
for (now_target, now_source) in all_ts_combination:
    print( "{} now_target:{}--now_source:{} {}".format("="*50, now_target, now_source, "="*50) )
    
    path_of_st = os.path.join(SAVE_DIR, f"S_{now_source}_T_{now_target}")
    if not os.path.exists(path_of_st): os.makedirs(path_of_st)
    
    df_s = all_dataset_f[now_source]
    df_t = all_dataset_f[now_target]
    
    user_set_s = set(df_s['reviewerID'])
    user_set_t = set(df_t['reviewerID'])
    
    share_user = user_set_s & user_set_t
    unshare_user_s = user_set_s - share_user
    unshare_user_t = user_set_t - share_user
    
    share_user_df_t = df_t[df_t['reviewerID'].isin(share_user)]
    share_user_df_t.to_csv(os.path.join(path_of_st, f'{now_target}_based_{now_source}_ratings.csv'),index=False)
    
    share_user_df_s = df_s[df_s['reviewerID'].isin(share_user)]
    share_user_df_s.to_csv(os.path.join(path_of_st, f'{now_source}_based_{now_target}_ratings.csv'),index=False)
    
    unshare_user_df_t = df_t[df_t['reviewerID'].isin(unshare_user_t)]
    unshare_user_df_t.to_csv(os.path.join(path_of_st, f'unshare_{now_target}_based_{now_source}_ratings.csv'),index=False)
    
    print(f"share_user:{len(share_user)}; unshare_user_s:{len(unshare_user_s)}; unshare_user_t:{len(unshare_user_t)};" +             f"\nshare_user_df_t:{share_user_df_t.shape}; share_user_df_s:{share_user_df_s.shape}; " +             f"unshare_user_df_t:{unshare_user_df_t.shape}")
    
    POOL_UNSHARE.append(unshare_user_df_t)

    
df_all = pd.concat(POOL_UNSHARE,)
df_all = df_all.drop_duplicates().reset_index(drop=True)
path_of_t_test = os.path.join(SAVE_DIR, f"{now_target}_test_data")
if not os.path.exists(path_of_t_test): os.makedirs(path_of_t_test)

user_index_array = df_all.groupby(['reviewerID']).size().index.values
item_index_array = df_all.groupby(['itemID']).size().index.values

user_existing_array ,user_new_array = train_test_split(user_index_array, test_size=0.2, random_state=42)
item_existing_array ,item_new_array = train_test_split(item_index_array, test_size=0.2, random_state=42)

Rw = df_all[(df_all['reviewerID'].isin(user_existing_array) & df_all['itemID'].isin(item_existing_array))]
Rw = get_convertdf(Rw)
trans_standart_test(Rw, save_name='Rw', path = path_of_t_test)
print(f"RW: {Rw.shape}")

Rci = df_all[(df_all['reviewerID'].isin(user_existing_array) & df_all['itemID'].isin(item_new_array))]
Rci = get_convertdf(Rci)
trans_standart_test(Rci, save_name='Rci', path = path_of_t_test)
print(f"Rci: {Rci.shape}")

Rcu = df_all[(df_all['reviewerID'].isin(user_new_array) & df_all['itemID'].isin(item_existing_array))]
Rcu = get_convertdf(Rcu)
trans_standart_test(Rcu, save_name='Rcu', path = path_of_t_test)
print(f"Rcu: {Rcu.shape}")

Rcui = df_all[(df_all['reviewerID'].isin(user_new_array) & df_all['itemID'].isin(item_new_array))]
Rcui = get_convertdf(Rcui)
trans_standart_test(Rcui, save_name='Rcui', path = path_of_t_test)
print(f"Rcui: {Rcui.shape}")



#######################################################################################################################

target_domain = ["book"]
source_domain = ['ele', 'musics', 'movies']
all_ts_combination = [[t, s] for s in source_domain for t in target_domain]
all_ts_combination = all_ts_combination
print(f"all_ts_combination: {all_ts_combination}")

print(f"Book\n\tTable shape:{all_dataset_f['book'].shape};  User num:{len(set(all_dataset_f['book']['reviewerID']))}"      + f";  item num:{len(set(all_dataset_f['book']['itemID']))}")

print(f"Cd\n\tTable shape:{all_dataset_f['cd'].shape};  User num:{len(set(all_dataset_f['cd']['reviewerID']))}"      + f";  item num:{len(set(all_dataset_f['cd']['itemID']))}")


FRAC_U = 0.2
FRAC_I = 1

tmp_book_u = list(set(all_dataset_f['book']['reviewerID']))
random.shuffle(tmp_book_u)
tmp_book_u = tmp_book_u[ : int(len(tmp_book_u)*FRAC_U)]

tmp_book_i = list(set(all_dataset_f['book']['itemID']))
random.shuffle(tmp_book_i)
tmp_book_i = tmp_book_i[ : int(len(tmp_book_i)*FRAC_I)]


tmp = all_dataset_f['book']
tmp = tmp[tmp['reviewerID'].isin(tmp_book_u) & tmp['itemID'].isin(tmp_book_i)].reset_index(drop=True)
# all_dataset_f['book'] = tmp

tmp = filterout(tmp, 5, 5)

print(f"Book\n\tTable shape:{tmp.shape}; "
      + f"User num:{len(set(tmp['reviewerID']))} "
      + f"item num:{len(set(tmp['itemID']))}")


all_dataset_f['book'] = tmp
# check
print(f"Book\n\tTable shape:{all_dataset_f['book'].shape};  User num:{len(set(all_dataset_f['book']['reviewerID']))}"      + f";  item num:{len(set(all_dataset_f['book']['itemID']))}")

print(f"Cd\n\tTable shape:{all_dataset_f['cd'].shape};  User num:{len(set(all_dataset_f['cd']['reviewerID']))}"      + f";  item num:{len(set(all_dataset_f['cd']['itemID']))}")


POOL_UNSHARE = []
for (now_target, now_source) in all_ts_combination:
    print( "{} now_target:{}--now_source:{} {}".format("="*50, now_target, now_source, "="*50) )
    
    path_of_st = os.path.join(SAVE_DIR, f"S_{now_source}_T_{now_target}")
    if not os.path.exists(path_of_st): os.makedirs(path_of_st)
    
    df_s = all_dataset_f[now_source]
    df_t = all_dataset_f[now_target]
    
    user_set_s = set(df_s['reviewerID'])
    user_set_t = set(df_t['reviewerID'])
    
    share_user = user_set_s & user_set_t
    unshare_user_s = user_set_s - share_user
    unshare_user_t = user_set_t - share_user
    
    share_user_df_t = df_t[df_t['reviewerID'].isin(share_user)]
    share_user_df_t.to_csv(os.path.join(path_of_st, f'{now_target}_based_{now_source}_ratings.csv'),index=False)
    
    share_user_df_s = df_s[df_s['reviewerID'].isin(share_user)]
    share_user_df_s.to_csv(os.path.join(path_of_st, f'{now_source}_based_{now_target}_ratings.csv'),index=False)
    
    unshare_user_df_t = df_t[df_t['reviewerID'].isin(unshare_user_t)]
    unshare_user_df_t.to_csv(os.path.join(path_of_st, f'unshare_{now_target}_based_{now_source}_ratings.csv'),index=False)
    
    print(f"share_user:{len(share_user)}; unshare_user_s:{len(unshare_user_s)}; unshare_user_t:{len(unshare_user_t)};" +             f"\nshare_user_df_t:{share_user_df_t.shape}; share_user_df_s:{share_user_df_s.shape}; " +             f"unshare_user_df_t:{unshare_user_df_t.shape}")
    
    POOL_UNSHARE.append(unshare_user_df_t)



    
df_all = pd.concat(POOL_UNSHARE,)
df_all = df_all.drop_duplicates().reset_index(drop=True)
path_of_t_test = os.path.join(SAVE_DIR, f"{now_target}_test_data")
if not os.path.exists(path_of_t_test): os.makedirs(path_of_t_test)

user_index_array = df_all.groupby(['reviewerID']).size().index.values

item_index_array = df_all.groupby(['itemID']).size().index.values

user_existing_array ,user_new_array = train_test_split(user_index_array, test_size=0.2, random_state=42)
item_existing_array ,item_new_array = train_test_split(item_index_array, test_size=0.2, random_state=42)



Rw = df_all[(df_all['reviewerID'].isin(user_existing_array) & df_all['itemID'].isin(item_existing_array))]
Rw = get_convertdf(Rw)
trans_standart_test(Rw, save_name='Rw', path = path_of_t_test)
print(f"RW: {Rw.shape}")


Rci = df_all[(df_all['reviewerID'].isin(user_existing_array) & df_all['itemID'].isin(item_new_array))]
Rci = get_convertdf(Rci)
trans_standart_test(Rci, save_name='Rci', path = path_of_t_test)
print(f"Rci: {Rci.shape}")


Rcu = df_all[(df_all['reviewerID'].isin(user_new_array) & df_all['itemID'].isin(item_existing_array))]
Rcu = get_convertdf(Rcu)
trans_standart_test(Rcu, save_name='Rcu', path = path_of_t_test)
print(f"Rcu: {Rcu.shape}")


Rcui = df_all[(df_all['reviewerID'].isin(user_new_array) & df_all['itemID'].isin(item_new_array))]
Rcui = get_convertdf(Rcui)
trans_standart_test(Rcui, save_name='Rcui', path = path_of_t_test)
print(f"Rcui: {Rcui.shape}")

