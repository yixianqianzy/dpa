
from model import kldLoss, ibLoss, miScore, MODEL 
import argparse
import torch
import torch.utils.data
from torch import nn, optim
import time
import os
import numpy as np
from dataset import Dataset
import itertools
import pandas as pd
from scipy.sparse import csr_matrix
import logging
from random import sample
from torchinfo import summary
from tqdm import tqdm
import pickle 
import random 


def set_global_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ArgParser():

    parser = argparse.ArgumentParser(description='MetaDPA')
    parser.add_argument('--method_name', type=str, default='MetaDPA', help='method_name')
    parser.add_argument('--gpu', type=str, default='0', help='select gpu id, -1 is not using')
    parser.add_argument('--print_on_screen', help='print_on_screen', type=bool, default=True)
    parser.add_argument('--model_summary', help='model_summary', type=bool, default=True)

    parser.add_argument('--prior', type=str, default='Gaussian', help='Gaussian, MVGaussian')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--batch', type=int, default=32, help='batch size.')
    parser.add_argument('--emb_size', type=int, default=400, help='embed size.')
    parser.add_argument('--mu_size', type=int, default=200, help='embed size.')
    parser.add_argument('--side_embed_size', type=int, default=300, help='side embed size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')

    parser.add_argument('--pos_weight', type=float, default=1.0, help='weight for positive samples')

    parser.add_argument('--reg', type=float, default=1.0, help='lambda reg')

    parser.add_argument('--jl_self', type=float, default=1.0, help='lambda rec')
    parser.add_argument('--jl_cross', type=float, default=1.0, help='lambda rec')
    parser.add_argument('--jl_ib', type=float, default=1.0, help='lambda rec')

    parser.add_argument('--savemodel', type=bool, default=True, help='save model?')

    parser.add_argument('--eta_jl', type=float, default=0.05, help='eta_jl')
    parser.add_argument('--eta_pl', type=float, default=1.0, help='eta_pl')
    parser.add_argument('--eta_cl', type=float, default=1.0, help='eta_cl')

    parser.add_argument('--me_cd', type=float, default=1.0, help='eta_cl')
    parser.add_argument('--me_di', type=float, default=1.0, help='eta_pl')
    parser.add_argument('--me_mi', type=float, default=1.0, help='eta_cl')

    parser.add_argument('--dataset', type=str, default='ele_cd', help='["ele_book", "movies_book", "musics_book", "ele_cd","movies_cd", "musics_cd"]')
    parser.add_argument('--reset_dataset', action='store_true', help='speedup')

    parser.add_argument('--datadir', type=str, default='../dataset/', help='')

    parser.add_argument('--debug', action='store_true', help='debug')

    parser.add_argument('--save', type=str, default='test', help='save')

    parser.add_argument('--generation', action='store_true', help='generation data? save time for me loss')

    parser.add_argument('--LOOP_NUMBER', type=int, default=50, help='LOOP_NUMBER')



    args = parser.parse_args()
    return args

def set_logger(args):
    log_file = os.path.join(args.save_path, 'train.log')
    if args.debug:
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.DEBUG,
            datefmt='%Y-%m-%d %H:%M:%S ',
            filename=log_file,
            filemode='w'
        )
    else:
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S ',
            filename=log_file,
            filemode='w'
        )

    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def main():
    args = ArgParser()
    args.log = 'logs/{}'.format(args.method_name)

    log = os.path.join(args.log, '{}_{}_{}'.format(args.dataset,  args.save, time.strftime("%m_%d")))
    
    if os.path.isdir(log):
        print("%s already exist. are you sure to override? Ok, I'll wait for 5 seconds. Ctrl-C to abort." % log)
        time.sleep(5)  # attention 
        os.system('rm -rf %s/' % log)
    
    args.save_path = log
    os.makedirs(log)
    set_logger(args)
    logging.info(args)
    logging.info("made the log directory: "+log)

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu   
    args.cuda = torch.cuda.is_available()

    logging.info('preparing data...')
    dataset = Dataset(args)

    if args.debug:
        args.epochs=5

    # domain A and domain B 
    NUM_USER = dataset.num_user
    NUM_SOURCE = dataset.num_source_item
    NUM_TARGET = dataset.num_target_item

    logging.info('Preparing the training data')
    # prepare data for X
    data, row, col = dataset.get_train_indices('source')
    values = data   # np.ones(row.shape[0])
    user_x = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_SOURCE)).toarray()

    # prepare  data fot Y
    data, row, col = dataset.get_train_indices('target')
    values = data   # np.ones(row.shape[0])
    user_y = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_TARGET)).toarray()

    logging.info('Finished preparing the training data')

    user_id = np.arange(NUM_USER).reshape([NUM_USER, 1])

    user_x = torch.FloatTensor(user_x)
    user_y = torch.FloatTensor(user_y)

    feat_x_item = {k:torch.FloatTensor(v) for k,v in dataset.source_item_feature.items()}
    feat_x_user = {k:torch.FloatTensor(v) for k,v in dataset.source_user_feature.items()}
    feat_y_item = {k:torch.FloatTensor(v) for k,v in dataset.target_item_feature.items()}
    feat_y_user = {k:torch.FloatTensor(v) for k,v in dataset.target_user_feature.items()}

    def _worker_init_fn_():
            torch_seed = torch.initial_seed()
            np_seed = torch_seed // 2**32-2
            np_seed = 0 if np_seed<0 else np_seed
            random.seed(torch_seed)
            np.random.seed(np_seed)
            
    train_loader = torch.utils.data.DataLoader(torch.from_numpy(user_id),
                                                     batch_size=args.batch,
                                                     num_workers=2, worker_init_fn=_worker_init_fn_(),
                                                     shuffle=True)


    pos_weight = torch.FloatTensor([args.pos_weight])

    if args.cuda:
        pos_weight = pos_weight.cuda()

    logging.info('Preparing model')

    model = MODEL(NUM_USER=NUM_USER, NUM_SOURCE=NUM_SOURCE, NUM_TARGET=NUM_TARGET,
                 EMBED_SIZE=args.emb_size, Z_DIM = args.mu_size, 
                 SIDE_INFO_USER=args.side_embed_size, SIDE_INFO_ITEM=args.side_embed_size, 
                 dropout=args.dropout)

    if args.debug:
        summary(model, input_data=[torch.randint(low=0, high=1, size=(2,NUM_SOURCE)).float(),
                            torch.randint(low=0, high=1, size=(2,NUM_TARGET)).float(),
                            torch.randint(low=0, high=10, size=(2,2000)).float(),
                            torch.randint(low=0, high=10, size=(2,2000)).float(),
                            torch.randint(low=0, high=10, size=(2,2000)).float(),
                            torch.randint(low=0, high=10, size=(2,2000)).float(),],
                        col_width=16,col_names=["kernel_size", "output_size", "num_params", "mult_adds"],)    

    BCEWL = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)
    MSE = torch.nn.MSELoss()
    KLD_Loss = kldLoss(Z_DIM = args.mu_size) 
    IB_Loss = ibLoss(Z_DIM = args.mu_size, EMBED_SIZE=args.emb_size)
    MI_Score = miScore(NUM_SOURCE=NUM_SOURCE, NUM_TARGET=NUM_TARGET)

    if args.cuda:
        model = model.cuda()
        KLD_Loss = KLD_Loss.cuda()
        IB_Loss = IB_Loss.cuda()
        MI_Score = MI_Score.cuda()

    for name, param in model.named_parameters():
        if 'side_info' in name:
            param.requires_grad = False
    optimizer_cd = optim.Adam( itertools.chain(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        KLD_Loss.parameters(),
                        IB_Loss.parameters(),
                        MI_Score.parameters()
                        ), lr=args.lr, weight_decay=args.weight_decay)    

    logging.debug("optimizer_cd: {} \n".format([i.shape for i in optimizer_cd.param_groups[0]['params']]))   # debug
        
    for name, param in model.named_parameters():
            param.requires_grad = True    

    
    optimizer_cl = optim.SGD( itertools.chain(
                    model.side_info_x_user_emb.parameters(),
                    model.side_info_x_encoder.parameters(),
                    model.side_info_y_user_emb.parameters(),
                    model.side_info_y_encoder.parameters()
                    ), lr=args.lr)     #lr=args.lr)

    logging.info('Finished Preparing model')


    Self_loss_list = []
    Cross_loss_list = []
    IB_Loss_list = []
    JL_loss_list = []
    PL_loss_list = []
    CL_loss_list = []
    CD_loss_list = []
    DI_loss_list = []
    MI_loss_list = []
    ME_loss_list = []
    CL_SI_loss_list = []

    for epoch in range(args.epochs):
        model.train()
        batch_Self_loss_list = []
        batch_Cross_loss_list = []
        batch_IB_Loss_list = []
        batch_JL_loss_list = []
        batch_PL_loss_list = []
        batch_CL_loss_list = []
        batch_CD_loss_list = []
        batch_MI_loss_list = []
        batch_ME_loss_list = []
        batch_CL_SI_loss_list = []

        # logging.info('==> {}'.format(epoch))
        for batch_idx, data in enumerate(train_loader):
            data = data.reshape([-1])
            LENGTH_USER = data.shape[0]
            # print(data.shape, data)

            batch_side_info_x_user = torch.cat([feat_x_user[x.item()].unsqueeze(0) for x in data],dim=0)
            batch_side_info_y_user = torch.cat([feat_y_user[x.item()].unsqueeze(0) for x in data],dim=0)
            batch_side_info_x_item = torch.zeros((LENGTH_USER, args.side_embed_size))
            batch_side_info_y_item = torch.zeros((LENGTH_USER, args.side_embed_size))

            batch_user_x = user_x[data]
            batch_user_y = user_y[data]
            batch_user = data
        

            for i, now_user in enumerate(batch_user_x):   
                idx = np.where(now_user>0)[0]
                # print(idx)
                batch_side_info_x_item[i]  = torch.stack([feat_x_item[idx_item.item()] for idx_item in idx], dim=0).mean(dim=0)

            for i, now_user in enumerate(batch_user_y):   
                idx = np.where(now_user>0)[0]
                # print(idx)
                batch_side_info_y_item[i]  = torch.stack([feat_y_item[idx_item.item()] for idx_item in idx], dim=0).mean(dim=0)

            if args.cuda:
                batch_user_x = batch_user_x.cuda()          # userid corresponding matrix 
                batch_user_y = batch_user_y.cuda()          # userid corresponding matrix 
                batch_side_info_x_user = batch_side_info_x_user.cuda()
                batch_side_info_y_user = batch_side_info_y_user.cuda()
                batch_side_info_x_item = batch_side_info_x_user.cuda()
                batch_side_info_y_item = batch_side_info_y_user.cuda()

            logging.debug(epoch, batch_idx, batch_user.shape, batch_user_x.shape, batch_user_y.shape, batch_side_info_x_user.shape, \
                    batch_side_info_y_user.shape, batch_side_info_x_item.shape, batch_side_info_y_item.shape)

            #############################################################################################################################
            # model
            #############################################################################################################################

            optimizer_cd.zero_grad()

            pred_x, pred_y, pred_x2y, pred_y2x, zmu_x, zmu_y, z_x, z_y, z_contro_x, z_contro_y, p_z_x, p_z_y, _, testgeneration = model(batch_user_x, batch_user_y, batch_side_info_x_user, \
                                                                                                                    batch_side_info_x_item, batch_side_info_y_user, batch_side_info_y_item, mode='all')
            
            
            # JL -> get plot JRL loss
            loss_x = BCEWL(pred_x, batch_user_x)
            loss_y = BCEWL(pred_y, batch_user_y)
            loss_x2y = BCEWL(pred_x2y, batch_user_y)
            loss_y2x = BCEWL(pred_y2x, batch_user_x)
            reg_loss = 0

            if args.jl_ib!=0:
                loss_ib = IB_Loss(p_z_x, p_z_y)    # MV InfoMax
            else:
                loss_ib = 0

            JL_loss = args.jl_self * (loss_x + loss_y) + args.jl_cross * (loss_x2y + loss_y2x) + args.jl_ib * loss_ib + args.reg*reg_loss
            batch_Self_loss_list.append((loss_x + loss_y).item()/args.batch)
            batch_Cross_loss_list.append((loss_x2y + loss_y2x).item()/args.batch)
            batch_IB_Loss_list.append(loss_ib.item()/args.batch) if args.jl_ib !=0 else batch_IB_Loss_list.append(0)
            batch_JL_loss_list.append(JL_loss.item()/args.batch)

            # PL -> get the PL loss  
            G_loss1 = KLD_Loss(zmu_x).sum()
            G_loss2 = KLD_Loss(zmu_y).sum()
            PL_loss = G_loss1+G_loss2                  
            batch_PL_loss_list.append(PL_loss.item()/args.batch)

            # CL -> get the  CL loss
            cl_x = MSE(z_contro_x, z_x)
            cl_y = MSE(z_contro_y, z_y)
            CL_loss = cl_x + cl_y
            batch_CL_loss_list.append(CL_loss.item()/args.batch)

            CD_loss  = args.eta_jl * JL_loss + args.eta_pl * PL_loss + args.eta_cl * CL_loss  # L_{CD}
            batch_CD_loss_list.append(CD_loss.item()/args.batch)

            #################################################### DI loss ####################################################
            # 原始网络计算
            mi_ya_pie, mi_ya, _ = model(batch_user_x, batch_user_y, batch_side_info_x_user, \
                              batch_side_info_x_item, batch_side_info_y_user, batch_side_info_y_item, mode='get_mi_yab')
            logging.debug('mi_ya_pie, mi_ya : {}   {}'.format(mi_ya_pie.shape , mi_ya.shape))

            #################################################### MI loss ####################################################
            # training network
            # mi_ya_pie, mi_ya 已经算过了
            ############################ 重采样生成y_hat ############################
            if args.me_mi!=0:
                # y_hat 新采样 x 通过 target domain 中的 decoder 生成的数据
                difference = list(set(user_id.flatten()) - set(data.numpy()))   # 差集
                try:
                    y_hat_data = sample(difference, LENGTH_USER)                    # unrelated user 
                except:
                    broadcast_factor = LENGTH_USER//len(difference) + 1
                    y_hat_data = sample(difference*broadcast_factor, LENGTH_USER)

                # 只需要 user id 和 对应的 feature
                batch_side_info_y_hat_user = torch.cat([feat_y_user[x.item()].unsqueeze(0) for x in y_hat_data],dim=0)
                batch_side_info_y_hat_item = torch.zeros((LENGTH_USER, args.side_embed_size))

                batch_y_hat_user_y = user_y[y_hat_data]                     # rating matrix
                batch_y_hat_user = torch.from_numpy(np.array(y_hat_data))   # user i -> embedding

                for i, now_user in enumerate(batch_y_hat_user_y):   
                    idx = np.where(now_user>0)[0]
                    batch_side_info_y_hat_item[i]  = batch_side_info_y_hat_item[i] + torch.stack([feat_y_item[idx_item.item()] for idx_item in idx], dim=0).sum(dim=0) 

                if args.cuda:
                    batch_side_info_y_hat_user = batch_side_info_y_hat_user.cuda()
                    batch_side_info_y_hat_item = batch_side_info_y_hat_item.cuda()

                ############################ model training ############################

                mi_ya_hat, _, _ = model(0, 0, 0, 0, batch_side_info_y_hat_user, batch_side_info_y_hat_item, mode='get_mi_yab')
                

                mi_score = MI_Score(mi_ya_pie, mi_ya )
                
                MI_loss = BCEWL(mi_score, mi_ya_hat.detach())
                
                del mi_score, mi_ya_hat

                batch_MI_loss_list.append(MI_loss.item()/args.batch)
            else:
                MI_loss = 0
                batch_MI_loss_list.append(0)

            ######################################################################################################################################

            ME_ALL_loss = args.me_cd * CD_loss + args.me_mi * MI_loss

            batch_ME_loss_list.append(ME_ALL_loss.item()/args.batch)

            optimizer_cd.zero_grad()
            ME_ALL_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer_cd.step()
            optimizer_cd.zero_grad()
            optimizer_cl.zero_grad()


            # 无论如何最后都要更新conditional那部分
            for loop_side in range(args.LOOP_NUMBER):
                z_x, z_y, z_contro_x, z_contro_y = model(batch_user_x, batch_user_y, batch_side_info_x_user, \
                                                        batch_side_info_x_item, batch_side_info_y_user, batch_side_info_y_item, mode='sideinfo')

                CL_FOR_SLIDEINFO_x = MSE(z_x*100, z_contro_x*100)   # MSE  check sum 
                CL_FOR_SLIDEINFO_y = MSE(z_y*100, z_contro_y*100)

                optimizer_cd.zero_grad()
                optimizer_cl.zero_grad()
                CL_SI_loss = CL_FOR_SLIDEINFO_x + CL_FOR_SLIDEINFO_y
                batch_CL_SI_loss_list.append(CL_SI_loss.item()/args.batch)
                CL_SI_loss.backward()
                # ipdb.set_trace()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)


                del z_x, z_y, z_contro_x, z_contro_y
                optimizer_cl.step()
                optimizer_cl.zero_grad()
                optimizer_cd.zero_grad()
                
        epoch_Self_loss = np.mean(batch_Self_loss_list)
        epoch_Cross_loss = np.mean(batch_Cross_loss_list)
        epoch_IB_Loss = np.mean(batch_IB_Loss_list)
        epoch_JL_loss = np.mean(batch_JL_loss_list)
        epoch_PL_loss = np.mean(batch_PL_loss_list)
        epoch_CL_loss = np.mean(batch_CL_loss_list)
        epoch_CD_loss = np.mean(batch_CD_loss_list)
        epoch_CL_SI_loss = np.mean(batch_CL_SI_loss_list)


        Self_loss_list.append(epoch_Self_loss)
        Cross_loss_list.append(epoch_Cross_loss)
        IB_Loss_list.append(epoch_IB_Loss)
        JL_loss_list.append(epoch_JL_loss)
        PL_loss_list.append(epoch_PL_loss)
        CL_loss_list.append(epoch_CL_loss)
        CD_loss_list.append(epoch_CD_loss)
        CL_SI_loss_list.append(epoch_CL_SI_loss)


        epoch_MI_loss = np.mean(batch_MI_loss_list)
        epoch_ME_loss = np.mean(batch_ME_loss_list)

        MI_loss_list.append(epoch_MI_loss)
        ME_loss_list.append(epoch_ME_loss)
        logging.info('epoch:{}/{}, Self loss:{:.4f}, Cross loss:{:.4f}, IB loss:{:.4f} --> JL loss:{:.4f},\n\t\t\t\t\t\t'
                    'PL loss:{:.4f}, CL loss:{:.4f}, --> CD loss:{:.4f}; CL4SI:{:.8f}, \n\t\t\t\t\t\t'
                    'MI loss:{:.4f}, ME loss:{:.4f} '.format(epoch, args.epochs,
                                            epoch_Self_loss, epoch_Cross_loss, epoch_IB_Loss, epoch_JL_loss,
                                            epoch_PL_loss, 
                                            epoch_CL_loss,
                                            epoch_CD_loss,
                                            epoch_CL_SI_loss,
                                            epoch_MI_loss, epoch_ME_loss,
                                            ))


    if args.savemodel:
        MODEL_SAVE_PATH = 'model_epoch_{}.pt'.format(epoch)
        torch.save(model.state_dict(), os.path.join(log,MODEL_SAVE_PATH))
        logging.info('Model successfully save! --> {}'.format(MODEL_SAVE_PATH))

    if args.generation:  
        SCALE_FACTOR = 2  # control the generation factor
        model.eval()
        with torch.no_grad():

            ANS = []     # pd.DataFrame(columns=['reviewerID', 'itemID', 'ratings'])
            POS_ANS = []
            NEG_ANS = []
            source_name = args.dataset.split('_')[0]  
            target_name = args.dataset.split('_')[1]

            # read unshare data
            Path_encoder1 = os.path.join(args.datadir, f"S_{source_name}_T_{target_name}", 'unshare_' + target_name + '_based_' + source_name + '_ratings.csv')    # unshare 
            logging.info("GENERATION ---> Processing unshare data: {}".format(Path_encoder1))
            rw_data1 = pd.read_csv(Path_encoder1)   
            if list(rw_data1.columns) != ['reviewerID', 'itemID', 'ratings']:
                rw_data1.columns = ['reviewerID', 'itemID', 'ratings']

            # read share data
            Path_encoder2 = os.path.join(args.datadir, f"S_{source_name}_T_{target_name}", target_name + '_based_' + source_name + '_ratings.csv')    # unshare 
            logging.info("GENERATION ---> Processing share data: {}".format(Path_encoder2))
            rw_data2 = pd.read_csv(Path_encoder2)   
            if list(rw_data2.columns) != ['reviewerID', 'itemID', 'ratings']:
                rw_data2.columns = ['reviewerID', 'itemID', 'ratings']

            rw_data = pd.concat([rw_data2, rw_data1]).reset_index(drop=True)
            logging.info("length of share data: {}, length of unshare data: {}, length of combinaton: {}".format( len(rw_data1), len(rw_data2), len(rw_data) ))

            # encod
            item2id = dict()
            user2id = dict()
            for idx, now_user in enumerate(set(rw_data['reviewerID'])):
                user2id[now_user] = idx
            for idx, now_item in enumerate(set(rw_data['itemID'])):
                item2id[now_item] = idx  

            item_name = np.array(list(dataset.target_item2id.keys()))  # 商品排序

            # unshared data
            feature_data_path = os.path.join("../dataset/tmp_word2vec_data", 'all_' + target_name + '_contentFeature.pkl')
            feature_data = pickle.load(open(feature_data_path, 'rb'))
            user_feature_dict = feature_data['user']
            item_feature_dict = feature_data['item']

            # logging.info("df-item_feature_data: {}, dict-item_feature_data: {}".format( len(item_feature_data), len(item_feature_dict) ))

            logging.info('Start Processing')
            # 一次一次的处理每一个用户
            for batch_idx, now_user in enumerate(tqdm(user2id.keys())):  # 遍历每一个用户  A2S166WSCFIFP5
                batch_user = torch.tensor(0) 
                batch_side_info_y_user = torch.tensor(user_feature_dict[now_user]).float().unsqueeze(0)

                now_user_all_rating_list = rw_data[rw_data['reviewerID']==now_user]['itemID'].tolist()
                length_select = len(now_user_all_rating_list)
                
                batch_side_info_y_item = torch.zeros((1, args.side_embed_size))
                batch_side_info_y_item[0]  = torch.stack([torch.tensor(item_feature_dict[item_code]) for item_code in now_user_all_rating_list], dim=0).mean(dim=0)

                if args.cuda:
                    batch_side_info_y_user = batch_side_info_y_user.cuda()
                    batch_side_info_y_item = batch_side_info_y_user.cuda()


                ans, z_contro_y = model(_, _, _, _, batch_side_info_y_user, batch_side_info_y_item, mode='generation')

                ans = ans.detach().cpu().numpy()[0]  
                now_order = np.argsort(ans)       
                now_item_name = item_name[now_order]

                if length_select/len(now_order)>0.02:
                    length_select = int(len(now_order)*0.01)

                mini_POS_ANS = [[now_user, now_item_name[-(i+1)], 1] for i in range(length_select * SCALE_FACTOR)]
                random.shuffle(mini_POS_ANS)
                POS_ANS = POS_ANS + mini_POS_ANS[ : int(len(mini_POS_ANS)/SCALE_FACTOR)]

                mini_NEG_ANS = [[now_user, now_item_name[i], 0] for i in range(length_select * 5 * SCALE_FACTOR)]
                random.shuffle(mini_NEG_ANS)
                NEG_ANS = NEG_ANS + mini_NEG_ANS[ : int(len(mini_NEG_ANS)/SCALE_FACTOR)]

                if batch_idx==100 and args.debug:   # only for test
                    break
            
            ANS = POS_ANS + NEG_ANS

            ANS = pd.DataFrame(ANS, columns=['reviewerID', 'itemID', 'ratings'])

            base_path = './generation'
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            ANS.to_csv( os.path.join(base_path, args.dataset + args.save + '_generation_rw.csv') ,index=0)
            
            logging.info("successfully saved!")

if __name__ == "__main__":
    set_global_seed()
    main()
    
