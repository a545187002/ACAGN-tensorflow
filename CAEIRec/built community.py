import networkx as nx
import pandas as pd
import tensorflow as tf
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import os
dij= np.loadtxt(open("dij_cuc.csv", "rb"), delimiter=",", skiprows=0)
inf=np.loadtxt(open("CUC.csv", "rb"), delimiter=",", skiprows=0)
ri_all = []
ri_index = []
def structural_interaction(ri_index, ri_all, g):
    """structural interaction between the communities"""
    for i in range(len(ri_index)):
        for j in range(len(ri_index)):
            intersection = set(ri_index[i]).intersection(set(ri_index[j]))
            union = set(ri_index[i]).union(set(ri_index[j]))
            intersection = list(intersection)
            union = list(union)
            intersection_ri_alli = []
            intersection_ri_allj = []
            union_ri_alli = []
            union_ri_allj = []
            g[i][j] = 0
            if len(intersection) == 0:
                g[i][j] = 0.0001
                continue
            else:
                union_rest = set(union).difference(set(intersection))
                union_rest = list(union_rest)
                print('intersection:' + str(len(intersection)) + 'union:' + str(len(union_rest)))
                if len(union_rest) == 0:
                    g[i][j] = 0.0001
                    continue
                else:
                    for k in range(len(union_rest)):
                        if union_rest[k] in ri_index[i]:
                            continue
                        else:
                            union_ri_allj.append(ri_all[j][ri_index[j].tolist().index(union_rest[k])])

                for k in range(len(intersection)):
                    intersection_ri_alli.append(ri_all[i][ri_index[i].tolist().index(intersection[k])])
                    intersection_ri_allj.append(ri_all[j][ri_index[j].tolist().index(intersection[k])])

                k_max = max(intersection_ri_allj, intersection_ri_alli)
                k_min = min(intersection_ri_allj, intersection_ri_alli)
                union_ri_allj = k_max + union_ri_allj
                union_num = np.sum(np.array(union_ri_allj), axis=0)
                inter_num = np.sum(np.array(k_min), axis=0)
                g[i][j] = inter_num / union_num
                print('i:'+str(i)+'\t'+'j:'+str(j))
        print("struct:"+str(i))

    return g
G = nx.DiGraph()
inf= np.loadtxt(open('data/UCU.csv', 'rb'), delimiter=",", skiprows=0)

for i in range(len(inf)):
    for j in range(len(inf[i])):
        if(inf[i][j]==1):
          G.add_edge(i, j, weight=1)
record=np.zeros((1944,1944))
for i in range(1944):
          for j in range(1944):
            if record[j][i]==1:
                inf[i][j]=inf[j][i]
            else:
              try:
                  rs = nx.astar_path_length \
                          (
                          G,
                          i,
                          j,
                      )
              except nx.exception.NodeNotFound :
                  rs=0
              except nx.exception.NetworkXNoPath:
                  rs=0
              if rs == 0:
                  length = 0
              else:
                  length = rs

              inf[i][j]=length
              record[i][j]=1
          print('dij'+str(i))
dij=inf
inf=pd.DataFrame(inf)
inf.to_csv('./data/dij_ucu.csv',index=False,header=False)
for i in range(1944):
            # You may replace 1,4 with the .n-hop neighbors you want
            index_i = np.where((dij[i]>0)&(dij[i]<=1))
            ei = []
            I = np.eye((len(index_i[0]) + 1), dtype=int)
            W=[]
            for j in range(len(index_i[0])+1):
                w=[]
                for k in range(len(index_i[0])+1):
                    if j==0:
                       if k==0:
                         w.append(float(0))
                       else:
                         w.append(float(inf[i][index_i[0][k-1]]))
                    else:
                       if k==0:
                         w.append(float(inf[index_i[0][j-1]][i]))
                       else:
                         w.append(float(inf[index_i[0][j-1]][k-1]))
                W.append(w)
            W=np.array(W)

            for m in range(W.shape[0]):
                count=0.0
                for n in range(W.shape[0]):
                    count+=W[n][m]
                for g in range(W.shape[0]):
                    if count!=0.0:
                     W[g][m]=W[g][m]/count





            for q in range((len(index_i[0]) + 1)):
                if q == 0:
                    ei.append([1])
                else:
                    ei.append([0])
            # W = []
            # for j in range((len(index_i[0])) + 1):
            #     w = []
            #     for k in range((len(index_i[0])) + 1):
            #         if j == 0:
            #             if k == 0:
            #                 w.append(float(0))
            #             else:
            #                 w.append(float(1))
            #         else:
            #             if k == 0:
            #                 w.append(float(1))
            #             else:
            #                 w.append(float(0))
            #     W.append(w)
            # the choice of the c parameter in RWR
            c = 0.5
            W = np.array(W)
            rw_left = (I - c * W)
            try:
                rw_left = np.linalg.inv(rw_left)
            except:
                rw_left = rw_left
            else:
                rw_left = rw_left
            ei = np.array(ei)
            rw_left = tf.tensor(rw_left, dtype=torch.float32)
            ei = tf.tensor(ei, dtype=torch.float32)
            ri = tf.matumul(rw_left, ei)
            ri = tf.transpose(ri, 1, 0)
            ri = abs(ri[0]).numpy().tolist()
            ri_index.append(index_i[0])
            ri_all.append(ri)
            print("rwr:"+str(i))
adj_delta = structural_interaction(ri_index, ri_all, inf)
adj_delta=pd.DataFrame(adj_delta)
adj_delta.to_csv("adj_delta_ucu_c0.5_k1_test1.csv",index=False,sep=',')
