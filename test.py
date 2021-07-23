
# import pandas as pd
# import torch.multiprocessing as mp
# import torch
# import torch.nn.functional as F
# import os
# from tqdm import tqdm
# import numpy as np
# from sklearn.preprocessing import minmax_scale 
# import pickle
# import torch.nn as nn
# from tqdm import tqdm
# import time 
# import copy
# import torch
# from torch.utils.data import DataLoader, random_split
# from MeLU import * 
# print("hello")
# class TheModelClass(nn.Module):
#     def __init__(self):
#         super(TheModelClass, self).__init__()
#         self.conv1 = nn.Conv2d(3, 8, 5)
#         self.bn = nn.BatchNorm2d(8)
#         self.conv2 = nn.Conv2d(8, 16, 5)
#         self.pool = nn.MaxPool2d(2, 2)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.bn(x)
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.flatten()[:5]
#         return x
    
#     # Initialize model

# def proc_eval_func(model, dataloader, ans):
#     for idx, batch in enumerate(tqdm(dataloader)):
#         out = model(torch.rand((1,3,24,24)).to('cuda:0'))
#         ans.append(out.cpu().detach().numpy().flatten())
#         time.sleep(0.05)
#     return ans 

# def chunks_list(lst, n):
#     division = len(lst) / float(n)
#     return [list(lst)[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]

# if __name__=='__main__':
#     model = TheModelClass().to('cuda:0')
    
#     group=5
#     dataset = list(range(978))
#     group_set = chunks_list(dataset,5)
#     # print(group_set)
#     dataset_group = random_split(dataset=dataset,lengths=[len(x) for x in group_set])
#     dataloader_test = [DataLoader(dataset_group[i], batch_size=1,num_workers=1) for i in range(len(group_set))]

#     # tmp = []
#     mp = mp.get_context("spawn")   
#     # pool = ctx.Pool(group)  # 这边设置多线程的线程数== 组大小
#     manager = mp.Manager()
#     pred_list = manager.list()
#     # for i in range(group):
#     #     process = ctx.Process(target = proc_eval_func, args=(model, dataloader_test[i], pred_list ))
#     #     process.start()
#     #     tmp.append(process)
#     # for p in pool:
#     #     p.join()  # 等待所有进程执行完毕
#     processes = []
#     for rank in range(5):
#         p = mp.Process(target=proc_eval_func, args=(model, dataloader_test[rank], pred_list))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()

#     print(pred_list)
#     # for i in processes:
#     #     data = i.get()
#     #     print(data)


# # import torch.multiprocessing as mp

# # def foo(n,L):
# #     L.append(n)

# # pool = mp.Pool(processes=2)
# # manager = mp.Manager()

# # L= manager.list()

# # l=[[1,2],[3,4],[5,6],[7,8]]

# # [pool.apply_async(foo, args=[n,L]) for n in l]
# # pool.close()
# # pool.join()
# # print(L)



# # import time
# # from tqdm import tqdm
# # import multiprocessing as mp

# # def func(mydict,mylist):
# #     mydict["index1"]="aaaa"
# #     mydict["index2"]="bbbb"
# #     mylist.append(11)
# #     mylist.append(22)
# #     mylist.append(33)


# # if __name__ == '__main__':
# #     with mp.Manager() as MG: #重命名
# #         mydict=MG.dict()#主进程与子进程共享这个字典
# #         mylist=MG.list(range(5))#主进程与子进程共享这个LIST

# #         p=mp.Process(target=func,args=(mydict,mylist))

# #         p.start()
# #         p.join()

# #         print(mylist)
# #         print(mydict)
