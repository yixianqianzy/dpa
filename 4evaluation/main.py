import copy
import os
import numpy as np
import torch
import torch.nn as nn 
from torch.nn import functional as F
from torch.utils.data import DataLoader
import util as utils
from dataset import Train_Dataset, Train_Dataset_abla, Test_Dataset
from logger import Logger
from model import selfmodel 
import argparse
import torch
import time
from tqdm import tqdm
import logging
from evaluation import evalutaion_topk 

def set_logger(save_path, name=None, print_on_screen=True):
    if name!=None:
        log_file = os.path.join(save_path, name+'.log')
    else:
        log_file = os.path.join(save_path, 'train.log')
    
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S ',
        filename=log_file,
        filemode='w'
    )
    if print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def ArgParser():
    parser = argparse.ArgumentParser([],description='MetaDPA')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='book')
    parser.add_argument('--reset_dataset', action='store_true', default=False, help='') 
    parser.add_argument('--supportdata', type=str, default='ele+movies+musics')
    
    parser.add_argument('--datadir', type=str, default='../dataset/', help='')
    parser.add_argument('--g_path', type=str, default='../3generation/generation', help='path to generation data root')

    parser.add_argument('--g_name', type=str, default='test', help='path to generation data name')

    parser.add_argument('--task', type=str, default='DPA', help='Label name')
    parser.add_argument('--tasks_per_metaupdate', type=int, default=64, help='number of tasks in each batch per meta-update')

    parser.add_argument('--lr_inner', type=float, default=0.01, help='inner-loop learning rate (per task)')
    parser.add_argument('--lr_meta', type=float, default=1e-3, help='outer-loop learning rate (used with Adam optimiser)')

    parser.add_argument('--num_grad_steps_inner', type=int, default=5, help='number of gradient steps in inner loop (during training)')
    parser.add_argument('--num_grad_steps_eval', type=int, default=1, help='number of gradient updates at test time (for evaluation)')

    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--test', action='store_true', default=False, help='num of workers to use')
    parser.add_argument('--debug', action='store_true', default=False, help='debug')
    parser.add_argument('--gpu', type=str, default="0", help='debug')
    parser.add_argument('--savename', type=str, default="test", help='save name')

    parser.add_argument('--num_epoch', type=int, default=30, help='num of epochs to use')
    
    parser.add_argument('--rerun', action='store_true', default=False,
                        help='Re-run experiment (will override previously saved results)')

    parser.add_argument('--model_summary', action='store_true', default=False, help='model_summary')
    parser.add_argument('--result_verbose', action='store_true', default=False, help='result_verbose')
    parser.add_argument('--abla_true_data', action='store_true', default=False, help='abla_sourcedomain_data')
    parser.add_argument('--featuremarker', type=str, default='_contentFeature.pkl', help='featuremarker')

    args = parser.parse_args()

    return args


def run(args, num_workers=1, log_interval=100, verbose=True, save_path=None):
    code_root = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir('{}/{}_result_files/'.format(code_root, args.task)):
        os.mkdir('{}/{}_result_files/'.format(code_root, args.task))

    if not args.abla_true_data:
        path = '{}/{}_result_files/'.format(code_root, args.task) + args.savename + '_' + args.g_name + '_' + time.strftime("%Y-%m-%d-%H-%M")
        set_logger(save_path='{}/{}_result_files/'.format(code_root, args.task), name=path.split('/')[-1])
    else:
        path = '{}/{}_result_files/'.format(code_root, args.task) + args.savename + '_abla_true_data_' + time.strftime("%Y-%m-%d-%H-%M")
        set_logger(save_path='{}/{}_result_files/'.format(code_root, args.task), name=path.split('/')[-1] + 'abla_true_data')        
    logging.info('File saved in ----> {}'.format(path))
    logging.info(args)


    if os.path.exists(path + '.pkl') and not args.rerun:
        logging.info('File has already existed. Try --rerun')
        exit()

    start_time = time.time()
    utils.set_seed(args.seed)



    # ---------------------------------------------------------
    # -------------------- training ---------------------------

    if args.debug:
        logging.info("epoch update to 3")
        args.num_epoch = 2
        args.tasks_per_metaupdate = 64

    # initialise model
    model = selfmodel(SIDE_INFO_ITEM=300, SIDE_INFO_USER=300, emb_size=128, fc_dim=64).cuda()

    if args.model_summary:
        from torchinfo import summary
        input_data = torch.randn(200, 4000)
        summary(model, input_data=input_data,col_width=16, \
                        col_names=["kernel_size", "output_size", "num_params", "mult_adds"],)

    model.train()
    logging.info("number of tersors in model: {}".format(sum([param.nelement() for param in model.parameters()])))  
    # set up meta-optimiser for model parameters
    meta_optimiser = torch.optim.Adam(model.parameters(), args.lr_meta)
    criterion_mse = nn.MSELoss()

    # initialise logger
    logger = Logger()
    logger.args = args

    # initialise the starting point for the meta gradient (it's faster to copy this than to create new object)
    meta_grad_init = [0 for _ in range(len(model.state_dict()))]
    if not args.abla_true_data:
        dataloader_train = DataLoader(Train_Dataset(args),
                                        batch_size=1,num_workers=args.num_workers, shuffle=True)
    else:
        dataloader_train = DataLoader(Train_Dataset_abla(args),
                                        batch_size=1,num_workers=args.num_workers, shuffle=True)        
    logging.info('Start training')
    for epoch in range(args.num_epoch):
        x_spt, y_spt, x_qry, y_qry = [],[],[],[]
        iter_counter = 0
        for step, batch in enumerate(dataloader_train):
            if len(x_spt)<args.tasks_per_metaupdate:   
                x_spt.append(batch[0][0].cuda())    
                y_spt.append(batch[1][0].cuda())    
                x_qry.append(batch[2][0].cuda())    
                y_qry.append(batch[3][0].cuda())   
                if not len(x_spt)==args.tasks_per_metaupdate:
                    continue
            
            if len(x_spt) != args.tasks_per_metaupdate:
                continue

            # initialise meta-gradient
            meta_grad = copy.deepcopy(meta_grad_init)
            loss_pre = []
            loss_after = []
            for i in range(args.tasks_per_metaupdate): 

                loss_pre.append(F.binary_cross_entropy(model(x_qry[i]), y_qry[i]).item())
                fast_parameters = model.final_part.parameters()
                for weight in model.final_part.parameters():
                    weight.fast = None
                for k in range(args.num_grad_steps_inner):
                    logits = model(x_spt[i])
                    # loss = F.mse_loss(logits, y_spt[i])
                    loss = F.binary_cross_entropy(logits, y_spt[i])
                    grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                    fast_parameters = []
                    for k, weight in enumerate(model.final_part.parameters()):
                        if weight.fast is None:
                            weight.fast = weight - args.lr_inner * grad[k]
                        else:
                            weight.fast = weight.fast - args.lr_inner * grad[k]  
                        fast_parameters.append(weight.fast)         

                logits_q = model(x_qry[i])

                loss_q = F.binary_cross_entropy(logits_q, y_qry[i])
                loss_after.append(loss_q.item())
                task_grad_test = torch.autograd.grad(loss_q, model.parameters())
                
                for g in range(len(task_grad_test)):
                    meta_grad[g] += task_grad_test[g].detach()
                    
            # -------------- meta update --------------
            
            meta_optimiser.zero_grad()

            # set gradients of parameters manually
            for c, param in enumerate(model.parameters()):
                param.grad = meta_grad[c] / float(args.tasks_per_metaupdate)
                param.grad.data.clamp_(-10, 10)

            # the meta-optimiser only operates on the shared parameters, not the context parameters
            meta_optimiser.step()
            x_spt, y_spt, x_qry, y_qry = [],[],[],[]
            loss_pre = np.array(loss_pre)
            loss_after = np.array(loss_after)
            logger.train_avg_loss.append(np.mean(loss_pre))
            logger.valid_avg_loss.append(np.mean(loss_after))
            logger.train_loss.append(loss_pre)
            logger.valid_loss.append(loss_after)
            logger.train_conf.append(1.96*np.std(loss_pre, ddof=0)/np.sqrt(len(loss_pre)))
            logger.valid_conf.append(1.96*np.std(loss_after, ddof=0)/np.sqrt(len(loss_after)))
            logger.test_avg_loss.append(np.mean(loss_after))
            logger.test_loss.append(0)
            logger.test_conf.append(0)
    
            utils.save_obj(logger, path)         # ??????????????????

            iter_counter += 1
            logger.print_info(epoch, iter_counter, start_time)
            start_time = time.time()

        if epoch!=0 or args.debug:
        # if True:
            logging.info('saving model at epoch {}'.format(epoch))
            logger.valid_model.append(copy.deepcopy(model))

            logging.info('='*50  + " EVAL " + '='*50)
            eval(args, model, logger, path, testdataset=['old', 'new_user', 'new_item', 'new_item_user'], partition='valid', name=f"{epoch}epoch", topk_list=[5, 20])
            eval(args, model, logger, path, testdataset=['old', 'new_user', 'new_item', 'new_item_user'], partition='test', name=f"{epoch}epoch", topk_list=[5, 20])
        
        if epoch%3==0:
            utils.save_obj(logger, path)         # ??????????????????

    return logger, model, path



def evaluate_test(args, model,  dataloader, logger=None, path=None):
    model.eval()
    loss_all = []
    ans = []

    for c, batch in enumerate(tqdm(dataloader)):

        x_spt = batch[0].cuda()  # shape -> torch.Size([1, 10, 1])
        y_spt = batch[1].cuda()
        x_qry = batch[2].cuda()
        y_qry = batch[3].cuda()

        for i in range(x_spt.shape[0]):
            # -------------- inner update --------------
            fast_parameters = model.final_part.parameters()
            for weight in model.final_part.parameters():
                weight.fast = None
            for k in range(args.num_grad_steps_inner):
                logits = model(x_spt[i])
                # loss = F.mse_loss(logits, y_spt[i])
                loss = F.binary_cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                fast_parameters = []
                for k, weight in enumerate(model.final_part.parameters()):
                    if weight.fast is None:
                        weight.fast = weight - args.lr_inner * grad[k] #create weight.fast 
                    else:
                        weight.fast = weight.fast - args.lr_inner * grad[k]  
                    fast_parameters.append(weight.fast)

            
            output =model(x_qry[i])
            # print(x_qry[i].shape, output.shape, y_qry[i].shape)
            loss_all.append(F.l1_loss(y_qry[i], output).item())

            ratings_true = y_qry[i].cpu().detach().numpy().flatten()
            ratings_pred = output.cpu().detach().numpy().flatten()
            ans.append([ratings_true,ratings_pred])

    loss_all = np.array(loss_all)
    logging.info('test way: {} --> MAE {}+/-{:.4f}; E\n'.format(   \
                args.test_way, np.mean(loss_all), 1.96*np.std(loss_all,0)/np.sqrt(len(loss_all))))

    return ans

def eval(args, model, logger=None, path=None, testdataset=['new_user', 'new_item', 'new_item_user', 'old'], \
            partition='test', name="", topk_list=[5,25]):

    for now_test_way in testdataset:
        logging.info('='*100)
        args.test_way = now_test_way
        dataloader_test = DataLoader(Test_Dataset(args, partition=partition, test_way=now_test_way),#old, new_user, new_item, new_item_user
                                    batch_size=1, num_workers=args.num_workers)
        pred_list = evaluate_test(args, model, dataloader_test)        # ??????????????????
        if args.result_verbose:
            _ = evalutaion_topk(pred_list, now_test_way, topk_list=topk_list)     # ????????????
        logger.ans[partition][name + now_test_way] = pred_list
    return None 


if __name__ == '__main__':
    args = ArgParser()
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
    logger, model, path = run(args, log_interval=100, verbose=True, save_path=None)
    logging.info("FINISHED!!!!!!!!!!")
    logging.info("EVAL!!!!!!!!!!")
    eval(args, model, logger, path)





