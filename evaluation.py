import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, auc, roc_curve,roc_auc_score # 均方误差
import math
import logging 
# def evaluate_ndcg(item_true, item_pred,top_k):
#     #注意y_pred、y_true是用户真实标签
#     df_pred = pd.DataFrame({"y_pred":item_pred})
#     df_real = pd.DataFrame({"y_true":item_true})
#     df_pred = df_pred.sort_values(by="y_pred", ascending=False)
#     df_real = df_real.sort_values(by="y_true", ascending=False)
#     df_pred = df_pred.iloc[0:top_k, :]  # 取前K个
#     df_real = df_real.iloc[0:top_k, :]
#     epsilon = 0.1**10
#     DCG = 0
#     iDCG = 0
#     positive_item =df_real.index
#     for i in range(top_k):
#         if df_pred.index[i] in positive_item:
#             DCG += 1 / np.log2(i + 2)
#     for i in range(min(len(positive_item), top_k)):
#         iDCG += 1 / np.log2(i + 2)
#     ndcg = DCG / max(iDCG, epsilon)
#     return ndcg

def evaluate_ndcg(item_true, item_pred,top_k):
    #注意y_pred、y_true是用户真实标签
    df_pred = item_pred[:top_k]  # 取前K个
    positive_item = item_true[:top_k]
    epsilon = 0.1**10
    DCG = 0
    iDCG = 0

    for i in range(top_k):
        if df_pred[i] in positive_item:
            DCG += 1 / np.log2(i + 2)
            # print('DCG is',DCG)
    for i in range(min(len(positive_item), top_k)):
        iDCG += 1 / np.log2(i + 2)
        # print('iDCG is',iDCG)
    ndcg = DCG / max(iDCG, epsilon)
    return ndcg

# def evaluate_accuracy(ratings_pred, ratings_true):
#     correct, sample_count = 0.0, 0.0
#     correct = float(sum(abs((ratings_pred - ratings_true)) <= 0.5))
#     sample_count = len(ratings_true)
#     return float(correct / sample_count)


# todo
# def evaluate_recall(item_true, item_pred, top_k):   # https://blog.csdn.net/qq_39490713/article/details/101711762
#
#     pred = item_pred[:top_k]
#     num_hit = len(set(pred).intersection(set(item_true)))    # 并集
#     print(pred, item_true, set(pred).intersection(set(item_true)))
#     # precision = float(num_hit) / len(pred)
#     recall = float(num_hit) / len(item_true)
#     return recall

def evaluate_hr(item_true,item_pred, top_k):
    item_true = item_true[:top_k]
    item_pred = item_pred[:top_k]
    count = 0
    for item in item_pred:
        if item in item_true:
            count += 1.0
    return count / len(item_true)

def evaluate_mrr(item_true, item_pred):
    all_score = []
    for pred_index, p in enumerate(item_pred):
        if p not in item_true:
            score = 0
        else:
            true_index = int(np.argwhere(item_true == p))   # 对应的index
            score = 1 if true_index <= pred_index else (1/(true_index-pred_index+1))
        all_score.append(score)
    return np.mean(all_score)

def evaluate_auc(ratings_true, ratings_pred):
    return roc_auc_score(ratings_true, ratings_pred)


def getHitRatio(ranklist, pred_true_no):
    return 1 if pred_true_no in ranklist[:,1] else 0


def getNDCG(ranklist, pred_true_no):
    if pred_true_no in ranklist[:,1]:
        return math.log(2) / math.log(np.where(ranklist[:,1]==pred_true_no)[0][0] + 2.0)
    else:
        return 0

def getMrr(ranklist, pred_true_no):
    if pred_true_no in ranklist[:,1]:
        return 1.0 / (np.where(ranklist[:,1]==pred_true_no)[0][0] + 1.0)
    else:
        return 0

def getMAP(ranklist, pred_true_no):
    m = len(ranklist)
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item[1] == pred_true_no:
            return 1 / ((i + 1.0) * m)
    return 0


def evaluate_all(ratings_true, ratings_pred, predictions, top_k=5):
    assert len(ratings_true) == len(ratings_pred)
    
    ratings_true = np.array(ratings_true) if isinstance(ratings_true, list) else ratings_true
    ratings_pred = np.array(ratings_pred) if isinstance(ratings_pred, list) else ratings_pred

    predictions_topk = predictions[:top_k,:]
    # print(predictions_topk)
    mse = mean_squared_error(ratings_true,ratings_pred)    # y_true, y_pred
    rmse = math.sqrt(mse)
    
    mae = mean_absolute_error(ratings_true,ratings_pred)    # y_true, y_pred
    
    hr = getHitRatio(predictions_topk,0)

    mrr = getMrr(predictions_topk,0)

    auc = evaluate_auc(ratings_true, ratings_pred)

    ndcg = getNDCG(predictions_topk, 0)

    MAP = getMAP(predictions_topk, 0)


    # print("mse:{:.4f}, mae:{:.4f}, hr:{:.4f}, mrr:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, auc:{:.4f}, ndcg:{:.4f}".format(mse, mae, hr, mrr, accuracy, recall, auc, ndcg ))
    # print("mae:{:.4f}, auc:{:.4f}, mrr:{:.4f}, hr:{:.4f}, recall:{:.4f}, ndcg:{:.4f}".format(mae, auc, mrr, hr, recall, ndcg ))
    return mse, rmse, mae, auc, mrr, hr, ndcg, MAP


# ans = evaluate_all(np.array([1,0,1,0,1]), np.array([0.2,0.9,0.8,0.2,0.6]), [1,2,3,4,5,6,7,8], [8,7,6,5,4,3,2,1], top_k=5)
# print(ans)


def evalutaion_topk(dataset, key, topk_list):
    ALL_SAVED_ANS = pd.DataFrame()
    
    for topk in topk_list:
        mse_all = []
        rmse_all = []
        mae_all = []
        auc_all = []
        mrr_all = []
        hr_all = []
        ndcg_all = []
        map_all = []

        idx_list = np.expand_dims(np.array(range(100)),1)
        for now_data in dataset:
            ratings_true, ratings_pred = now_data
            ratings_true = ratings_true.reshape(-1,100)  # check
            ratings_pred = ratings_pred.reshape(-1,100)

            """
            predictions:shape is (100, 2)   the first column is the probability of an item, the second column is the index
            ( suppose the true item is 0 and negative items are [1,2,3,...,99] )
            np.array([[0.7,99], (negative item)
                    [0.6,81], (negative item)
                    [0.5,0 ], (positive item)
                    ......
                        ])
            """
            for idx in range(ratings_true.shape[0]):
                # evaluate_all(ratings_true, ratings_pred, predictions, top_k=5)
                predictions = np.expand_dims(ratings_pred[idx], 1)
                predictions = np.concatenate((predictions,idx_list),axis=1)
                predictions = predictions[np.argsort(predictions[:,0])[::-1]]  # 100x2
                now_mse, now_rmse, now_mae, now_auc, now_mrr, now_hr, now_ndcg, now_map = evaluate_all(ratings_true[idx], ratings_pred[idx], predictions, top_k=topk)
                mse_all.append(now_mse)
                rmse_all.append(now_rmse)
                mae_all.append(now_mae)
                auc_all.append(now_auc)   
                mrr_all.append(now_mrr)
                hr_all.append(now_hr)
                ndcg_all.append(now_ndcg)
                map_all.append(now_map)

        logging.info('DATA: {} TOPK:{} \n\t -> Evaluation MAE:{}, AUC:{:.4f}, MRR@{}:{:.4f}, HR@{}:{:.4f}, NDCG@{}:{:.4f}, MAP@{}:{:.4f}\n'.format(   \
            key, topk, np.mean(mae_all), np.mean(auc_all), topk, np.mean(mrr_all), topk, np.mean(hr_all), topk, np.mean(ndcg_all), topk, np.mean(map_all)))
    return np.mean(mse_all), np.mean(rmse_all), np.mean(mae_all), np.mean(auc_all), np.mean(mrr_all), np.mean(hr_all), np.mean(ndcg_all), np.mean(map_all)
