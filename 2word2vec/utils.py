import numpy as np
import pandas as pd
import re
import random as rd
import os 

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
    before = len(df)
    
    index = df[["overall", "asin"]].groupby('asin').count() >= min_u
    item = set(index[index['overall'] == True].index)
    
    index = df[["overall", "reviewerID"]].groupby('reviewerID').count() >= min_i
    user = set(index[index['overall'] == True].index)
    
    df = df[df['asin'].isin(item)]
    df = df[df['reviewerID'].isin(user)]
    
    after = len(df)
    print(f"before:{before}-after:{after} = {after/before}")
    return df

def dataset_filtering(interaction, core):
    # filtering the dataset with core
    user_id_dic = {}  # record the number of interaction for each user and item
    item_id_dic = {}
    for [user_id, item_id, _] in interaction:
        try:
            user_id_dic[user_id] += 1
        except:
            user_id_dic[user_id] = 1
        try:
            item_id_dic[item_id] += 1
        except:
            item_id_dic[item_id] = 1
    print ('#Original dataset')
    print ('  User:', len(user_id_dic), 'Item:', len(item_id_dic), 'Interaction:', len(interaction), 'Sparsity:', 100 - len(interaction) * 100.0 / len(user_id_dic) / len(item_id_dic), '%')
    sort_user = []
    sort_item = []
    for user_id in user_id_dic:
        sort_user.append((user_id, user_id_dic[user_id]))  # id 出现次数
    for item_id in item_id_dic:
        sort_item.append((item_id, item_id_dic[item_id]))  # id 出现次数
    sort_user.sort(key=lambda x: x[1])
    sort_item.sort(key=lambda x: x[1])
    print ('Fitering(core = ', core, '...', end = '')

    while sort_user[0][1] < core or sort_item[0][1] < core:
        # find out all users and items with less than core recorders
        user_LessThanCore = set()
        item_LessThanCore = set()
        for pair in sort_user:
            if pair[1] < core:
                user_LessThanCore.add(pair[0])
            else:
                break
        for pair in sort_item:
            if pair[1] < core:
                item_LessThanCore.add(pair[0])
            else:
                break

        # reconstruct the interaction record, remove the cool one
        interaction_filtered = []
        for [user_id, item_id, text] in interaction:
            if not (user_id in user_LessThanCore or item_id in item_LessThanCore):
                interaction_filtered.append([user_id, item_id, text])
        # update the record
        interaction = interaction_filtered
        

        # count the number of each user and item in new data, check if all cool users and items are removed
        # reset all memory variables
        user_id_dic = {}  # record the number of interaction for each user and item
        item_id_dic = {}
        for [user_id, item_id, _] in interaction:
            try:
                user_id_dic[user_id] += 1
            except:
                user_id_dic[user_id] = 1
            try:
                item_id_dic[item_id] += 1
            except:
                item_id_dic[item_id] = 1

        sort_user = []
        sort_item = []
        for user_id in user_id_dic:
            sort_user.append((user_id, user_id_dic[user_id]))
        for item_id in item_id_dic:
            sort_item.append((item_id, item_id_dic[item_id]))
        sort_user.sort(key=lambda x: x[1])
        sort_item.sort(key=lambda x: x[1])
        print (len(interaction), end = ' ')
    print()
    print ('#Filtered dataset')
    print ('  User:', len(user_id_dic), 'Item:', len(item_id_dic), 'Interaction:', len(interaction), 'Sparsity:', 100 - len(interaction) * 100.0 / len(user_id_dic) / len(item_id_dic), '%')
    return interaction


def clean_string(string):
    string = re.sub(r"\'s", " is", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " had", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"[^A-Za-z]", " ", string)
    return string.strip().lower()

def index_encoding(interaction):
    # mapping in into number
    # after filtering the dataset, we need to re-encode the index of users and items
    user_id_set = set()
    item_id_set = set()
    for [user_id, item_id, _] in interaction:
        user_id_set.add(user_id)
        item_id_set.add(item_id)
    user_num2id = list(user_id_set)
    item_num2id = list(item_id_set)
    user_num2id.sort()
    item_num2id.sort()
    user_num = len(user_num2id)
    item_num = len(item_num2id)
    # user_id2num maps id to number, and user_num2id dictionary is not needed, user_ID
    user_id2num = {}
    for num in range(user_num):
        user_id2num[user_num2id[num]] = num
    item_id2num = {}
    for num in range(item_num):
        item_id2num[item_num2id[num]] = num
    interaction_num = []
    user_text = [set() for x in range(user_num)]
    item_text = [set() for x in range(item_num)]
    text = []
    for [user_id, item_id, review_text] in interaction:
        interaction_num.append([user_id2num[user_id], item_id2num[item_id]])
        user_text[user_id2num[user_id]] = user_text[user_id2num[user_id]] | review_text
        item_text[item_id2num[item_id]] = item_text[item_id2num[item_id]] | review_text
        text += list(review_text)
    text = set(text)

    num2word = list(text)
    word2num = {}
    for (i, word) in enumerate(num2word):
        word2num[word] = i
    user_text_num = [[] for x in range(user_num)]
    for u in range(user_num):
        user_text_num[u] = list(map(lambda x: word2num[x], list(user_text[u])))
    item_text_num = [[] for x in range(item_num)]
    for i in range(item_num):
        item_text_num[i] = list(map(lambda x: word2num[x], list(item_text[i])))
    return interaction_num, user_text_num, item_text_num, word2num, user_id2num, item_id2num


def semantic_embedding(word2num, path):
    matrix = np.random.uniform(-1.0, 1.0, (len(word2num), 300))
    with open(path, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('utf-8', 'ignore')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in word2num:
                matrix[word2num[word]] = np.fromstring(f.read(binary_len), dtype='float32')
            else: f.read(binary_len)
    return matrix


def semantic_embedding(word2num, path):
    matrix = np.random.uniform(-1.0, 1.0, (len(word2num), 300))
    with open(path, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('utf-8', 'ignore')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in word2num:
                matrix[word2num[word]] = np.fromstring(f.read(binary_len), dtype='float32')
            else: f.read(binary_len)
    return matrix


def dataset_sparse(interaction):
    rd.shuffle(interaction)
    train_interaction = interaction
    user_num = 0
    for [user, item ] in interaction:
        user_num = max(user_num, user)
    user_num += 1
    train_data = [[] for x in range(user_num)]
    for [user, item] in train_interaction:
        train_data[user].append(item)
    return train_data

# def dataset_split_sparse(interaction):
#     rd.shuffle(interaction)
#     n = int(len(interaction) * 0.1)
#     test_interaction = interaction[0: n]
#     validation_interaction = interaction[n: 2*n]
#     train_interaction = interaction[2*n: -1]
#     user_num = 0
#     for [user, item ] in interaction:
#         user_num = max(user_num, user)
#     user_num += 1
#     train_data = [[] for x in range(user_num)]
#     test_data = [[] for x in range(user_num)]
#     validation_data = [[] for x in range(user_num)]
#     for [user, item] in train_interaction:
#         train_data[user].append(item)
#     for [user, item] in test_interaction:
#         test_data[user].append(item)
#     for [user, item] in validation_interaction:
#         validation_data[user].append(item)
#     return train_data, validation_data, test_data