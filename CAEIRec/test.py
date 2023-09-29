
import pickle as pkl
import random

import networkx as nx
import numpy as np
import pandas as pd


# def radom_negative_sample(user_action,item_size):
#     negative_sample = []
#     for u in user_action:
#         sample = []
#         i = 0
#         while i < 99:
#             t = random.randint(0,item_size-1)
#             if t not in user_action[u]:
#                 sample.append([u, t])
#                 i += 1
#         sample.append([u, user_action[u][-1]])
#         negative_sample.append(sample)
#     return np.array(negative_sample)
#
#
# dict={}
# rate_matrix = np.loadtxt(open("./data1/rate_matrix.csv","rb"),delimiter=",",skiprows=0)
# for row_idx,row in enumerate(rate_matrix):
#
#     for col_idx,element in enumerate(row):
#         if element==1:
#            dict.setdefault(row_idx,[]).append(col_idx)
#
# negative=radom_negative_sample(dict,706)
#
# print(negative.shape)
# with open('./data/user_action.p', 'rb') as source:
#     user_action = pkl.load(source)
#     print(radom_negative_sample(user_action,5000).shape)
def dijskra(adj):
    G = nx.DiGraph()
    for i in range(len(adj)):
        for j in range(len(adj[i])):
            G.add_edge(i, adj[i][j], weight=1)
    for i in range(len(adj)):
        for j in range(len(adj[i])):
            try:
                rs = nx.astar_path_length \
                        (
                        G,
                        i,
                        j,
                    )
            except nx.NetworkXNoPath:
                rs = 0
            if rs == 0:
                length = 0
            else:
                length = rs
                if length>1:
                    print(length)
            adj[i][j] = length
    return  adj
ucu = np.loadtxt(open("./data1/UCU.csv", "rb"), delimiter=",", skiprows=0)
a=dijskra(ucu)
print(a)
