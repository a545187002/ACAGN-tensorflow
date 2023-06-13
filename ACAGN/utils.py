import networkx as nx
import numpy as np
import pickle as pkl
import random
import scipy.sparse as sp
import json

# load data and preprocessing
def load_data(user,item):
    support_user = []
    support_item = []
    dijskra_user = []
    dijskra_item = []
    # rating matrix

    rating = np.loadtxt(open("./data2/rate_matrix.csv", "rb"), delimiter=",",skiprows=0)

    # course_w2v = np.loadtxt(open("./data2/course_fea.csv", "rb"), skiprows=0)
    # course_bow = np.loadtxt(open("./data2/CV.csv", "rb"), delimiter=",", skiprows=0)
    course = np.loadtxt(open("./data2/course_fea.csv", "rb"), skiprows=0)
    features_course = preprocess_features(course.astype(np.float32))
    # user features
    # user_w2v = np.loadtxt(open("./data3/u_fea.csv", "rb"), delimiter=",", skiprows=0)
    # user_bow = np.loadtxt(open("./data3/UK.csv", "rb"), delimiter=",", skiprows=0)
    user_fea = np.loadtxt(open("./data2/u_fea.csv", "rb"), delimiter=",", skiprows=0)
    features_user = preprocess_features(user_fea.astype(np.float32))

    # uku
    if 'uku' in user:
        uku = np.loadtxt(open("./data2/UKU.csv", "rb"), delimiter=",", skiprows=0)

        support_user.append(uku)
        dijskra_user.append(dijskra(uku))
    # ucu
    if 'ucu' in user:
        ucu = np.loadtxt(open("./data2/UCU.csv", "rb"), delimiter=",", skiprows=0)
        support_user.append(ucu)
        dij_ucu = np.loadtxt(open("./data2/adj_delta_ucu_c0.5_k1_test.csv", "rb"), delimiter=",", skiprows=0)
        dijskra_user.append(dij_ucu)

    if 'uiu' in user:
        uiu = np.loadtxt(open("./data3/UIU.csv", "rb"), delimiter=",", skiprows=0)
        support_user.append(uiu)
        dij_uiu = np.loadtxt(open("./data3/dj_delta_uiu_c0.5_k1_test.csv", "rb"), delimiter=",", skiprows=0)
        dijskra_user.append(dij_uiu)

    if 'uiciu' in user:
        uiciu = np.loadtxt(open("./data3/UICIU.csv", "rb"), delimiter=",", skiprows=0)
        support_user.append(uiciu)
        dij_uiciu = np.loadtxt(open("./data3/dj_delta_uiciu_c0.5_k1_test.csv", "rb"), delimiter=",", skiprows=0)
        dijskra_user.append(dij_uiciu)



    # uvu
    if 'uvu' in user:
        uvu = np.loadtxt(open("./data2/UVU.csv", "rb"), delimiter=",", skiprows=0)

        support_user.append(uvu)
        dij_uvu= np.loadtxt(open("./data2/adj_delta_uvu_c0.5_k4_test.csv", "rb"), delimiter=",", skiprows=0)
        dijskra_user.append(dij_uvu)
    # uctcu
    if 'uctcu' in user:
        uctcu = np.loadtxt(open("./data2/UCTCU.csv", "rb"), delimiter=",", skiprows=0)

        support_user.append(uctcu)
        dij_uctcu = np.loadtxt(open("./data2/adj_delta_uctcu_c0.5_k3_test.csv", "rb"), delimiter=",", skiprows=0)
        dijskra_user.append(dij_uctcu)
    # ckc
    if 'ckc' in item:
        ckc = np.loadtxt(open("./data2/CKC.csv", "rb"), delimiter=",", skiprows=0)
        support_item.append(ckc)
        dij_ckc= np.loadtxt(open("./data2/adj_delta_ckc_c0.5_k3_test.csv", "rb"), delimiter=",", skiprows=0)
        dijskra_item.append(dij_ckc)
    # cuc
    if 'cuc' in item:
        cuc = np.loadtxt(open("./data2/CUC.csv", "rb"), delimiter=",", skiprows=0)
        support_item.append(cuc)
        dij_cuc = np.loadtxt(open("./data2/adj_delta_cuc_c0.5_k1_test.csv", "rb"), delimiter=",", skiprows=0)
        dijskra_item.append(dij_cuc)

    if 'ici' in item:
        ici = np.loadtxt(open("./data3/ICI.csv", "rb"), delimiter=",", skiprows=0)
        support_item.append(ici)
        dij_ici = np.loadtxt(open("./data3/dj_delta_ici_c0.5_k2_test.csv", "rb"), delimiter=",", skiprows=0)
        dijskra_item.append(dij_ici)

    if 'iui' in item:
        iui = np.loadtxt(open("./data3/IUI.csv", "rb"), delimiter=",", skiprows=0)
        support_item.append(iui)
        dij_iui = np.loadtxt(open("./data3/dj_delta_iui_c0.5_k1_test.csv", "rb"), delimiter=",", skiprows=0)
        dijskra_item.append(dij_iui)

    # negative sample
    dict = {}
    rate_matrix = np.loadtxt(open("./data2/test_rate_matrix.csv", "rb"), delimiter=",", skiprows=0)
    for row_idx, row in enumerate(rate_matrix):
        for col_idx, element in enumerate(row):
            if element == 1:
                dict.setdefault(row_idx, []).append(col_idx)
    negative = radom_negative_sample(dict, 694)
    support_user = np.array(support_user)
    support_item = np.array(support_item)
    dijskra_user = np.array(dijskra_user)
    dijskra_item = np.array(dijskra_item)
    return rating, features_course, features_user, support_user, support_item, negative,dijskra_user,dijskra_item


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return features

def preprocess_adj(adjacency):
    rowsum = np.array(adjacency.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adjacency.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)*1e2

def construct_feed_dict(placeholders, features_user, features_item, rating, biases_list_user,
                        biases_list_item, negative):
    feed_dict = dict()
    feed_dict.update({placeholders['rating']: rating})
    feed_dict.update({placeholders['features_user']: features_user})
    feed_dict.update({placeholders['features_item']: features_item})
    feed_dict.update({placeholders['support_user'][i]: biases_list_user[i] for i in range(len(biases_list_user))})
    feed_dict.update({placeholders['support_item'][i]: biases_list_item[i] for i in range(len(biases_list_item))})
    feed_dict.update({placeholders['negative']: negative})
    return feed_dict

def radom_negative_sample(user_action,item_size):
    negative_sample = []
    for u in user_action:
        sample = []
        i = 0
        while i < 99:
            t = random.randint(0,item_size-1)
            if t not in user_action[u]:
                sample.append([u, t])
                i += 1
        sample.append([u, user_action[u][-1]])
        negative_sample.append(sample)
    return np.array(negative_sample)

def getRateMatrix(user_action,item_size):
    row = []
    col = []
    dat = []
    for u in user_action:
        ls = set(user_action[u])
        for k in ls:
            row.append(u)
            col.append(k)
            dat.append(user_action[u].count(k))
    coo_matrix = sp.coo_matrix((dat,(row,col)),shape=(len(user_action),item_size))
    with open('./data/rate_matrix_new.p','wb') as source:
        pkl.dump(coo_matrix.toarray(),source)

def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):

            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
def dijskra(adj):
    G = nx.DiGraph()
    for i in range(len(adj)):
        for j in range(len(adj[i])):
            G.add_edge(i, adj[i][j], weight=1)
    for i in range(len(adj)):
        for j in range(len(adj)):
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
            adj[i][j] = length
    return  adj
def structural_interaction(ri_index, ri_all, g):
    """structural interaction between the structural fingerprints for citeseer"""
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
                break
            else:
                for k in range(len(intersection)):
                    intersection_ri_alli.append(ri_all[i][ri_index[i].tolist().index(intersection[k])])
                    intersection_ri_allj.append(ri_all[j][ri_index[j].tolist().index(intersection[k])])
                union_rest = set(union).difference(set(intersection))
                union_rest = list(union_rest)
                if len(union_rest) == 0:
                    g[i][j] = 0.0001
                    break
                else:
                    for k in range(len(union_rest)):
                        if union_rest[k] in ri_index[i]:
                            union_ri_alli.append(ri_all[i][ri_index[i].tolist().index(union_rest[k])])
                        else:
                            union_ri_allj.append(ri_all[j][ri_index[j].tolist().index(union_rest[k])])
                k_max = max(intersection_ri_allj, intersection_ri_alli)
                k_min = min(intersection_ri_allj, intersection_ri_alli)
                union_ri_allj = k_max + union_ri_allj
                union_num = np.sum(np.array(union_ri_allj), axis=0)
                inter_num = np.sum(np.array(k_min), axis=0)
                g[i][j] = inter_num / union_num

    return g