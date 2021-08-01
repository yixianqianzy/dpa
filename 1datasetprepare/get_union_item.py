# target domain union item 

import pandas as pd
import pickle
import os 

BASE_DIR = "../dataset"

ans={}

for now_target in [['S_ele_T_book','S_movies_T_book','S_musics_T_book'],['S_ele_T_cd','S_movies_T_cd','S_musics_T_cd']]:
    target_name = now_target[0].split('_')[-1]

    source_name_list = [now_target[x].split('_')[1] for x in range(3) ]

    print(f'{BASE_DIR}/{now_target[0]}/{target_name}_based_{source_name_list[0]}_ratings.csv')

    d0 = pd.read_csv(f'{BASE_DIR}/{now_target[0]}/{target_name}_based_{source_name_list[0]}_ratings.csv')
    d1 = pd.read_csv(f'{BASE_DIR}/{now_target[1]}/{target_name}_based_{source_name_list[1]}_ratings.csv')
    d2 = pd.read_csv(f'{BASE_DIR}/{now_target[2]}/{target_name}_based_{source_name_list[2]}_ratings.csv')

    print(d0.columns, d1.columns, d2.columns,)
    union_item = list(set(d0["itemID"]) | set(d1["itemID"]) | set(d2["itemID"]))
    print("==> item union length: {}".format(len(union_item)))  # 并集

    item2id={}
    for idx, now_item in enumerate(union_item):
        item2id[now_item] = idx

    if 'book' in now_target[0]:
        ans['book'] = item2id
    else:
        ans['cd'] = item2id

pickle.dump(ans, open( os.path.join(BASE_DIR, 'all_target_domain_item_dict.pkl'), 'wb'))