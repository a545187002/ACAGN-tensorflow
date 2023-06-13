import tensorflow as tf

import base_gattn
from layers import *
from metrics import *
import tensorflow as tf
import layers
from base_gattn import BaseGAttN
flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.test = None
        self.alphas = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for i in range(len(self.layers)):
            hidden = self.layers[i](self.activations[-1])
            if i == 3:
                # self.test = self.layers[i].test
                self.test = hidden
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        self._loss()

    def _loss(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class HeteGAT_multi(BaseGAttN):

    def inference(self,inputs_list, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat_list,adj_list,hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=16):
        embed_list = []
        for inputs, bias_mat,adj in zip(inputs_list, bias_mat_list,adj_list):
            attns = []
            jhy_embeds = []
            for _ in range(n_heads[0]):

                attns.append(layers.attn_head(inputs, out_sz=hid_units[0],bias_mat=bias_mat,adj=adj,
                                               activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=False))



            h_1 = tf.concat(attns, axis=-1)

            # for i in range(1, len(hid_units)):
            #     h_old = h_1
            #     attns = []
            #     for _ in range(n_heads[i]):
            #         attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
            #                                       out_sz=hid_units[i],
            #                                       activation=activation,
            #                                       in_drop=ffd_drop,
            #                                       coef_drop=attn_drop, residual=residual))
            #
            #     h_1 = tf.concat(attns, axis=-1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))

        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        print(mp_att_size)


        return  final_embed
    def inference1(self,inputs_list, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat_list,adj_list,hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=16):
        embed_list = []
        count=0
        for inputs, bias_mat,adj in zip(inputs_list, bias_mat_list,adj_list):
            attns = []
            jhy_embeds = []
            if count==0:
             for _ in range(n_heads[0]):

                attns.append(layers.attn_head(inputs, out_sz=hid_units[0],bias_mat=bias_mat,adj=adj,
                                               activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=False))



             h_1 = tf.concat(attns, axis=-1)

            # for i in range(1, len(hid_units)):
            #     h_old = h_1
            #     attns = []
            #     for _ in range(n_heads[i]):
            #         attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
            #                                       out_sz=hid_units[i],
            #                                       activation=activation,
            #                                       in_drop=ffd_drop,
            #                                       coef_drop=attn_drop, residual=residual))
            #
            #     h_1 = tf.concat(attns, axis=-1)
             embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))
             count+=1
            elif count==1:
             for _ in range(n_heads[0]):

                attns.append(layers.attn_head1(inputs, out_sz=hid_units[0],bias_mat=bias_mat,
                                               activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=False))



             h_1 = tf.concat(attns, axis=-1)

            # for i in range(1, len(hid_units)):
            #     h_old = h_1
            #     attns = []
            #     for _ in range(n_heads[i]):
            #         attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
            #                                       out_sz=hid_units[i],
            #                                       activation=activation,
            #                                       in_drop=ffd_drop,
            #                                       coef_drop=attn_drop, residual=residual))
            #
            #     h_1 = tf.concat(attns, axis=-1)
             embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))


        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)


        return  final_embed
# class HeteGAT_multi(BaseGAttN):  #maxpooling
#     def inference(self,inputs_list, nb_nodes, training, attn_drop, ffd_drop,
#                   bias_mat_list,adj_list,hid_units, n_heads,dim,activation=tf.nn.elu, residual=False,
#                   mp_att_size=128):
#         embed_list = []
#         for inputs, bias_mat,adj in zip(inputs_list, bias_mat_list,adj_list):
#             attns = []
#             jhy_embeds = []
#             for _ in range(n_heads[0]):
#                 attns.append(layers.attn_head(inputs, out_sz=hid_units[0], bias_mat=bias_mat, adj=adj,
#                                                                                               activation=activation,
#                                                                                            in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
#
#             h_1 = tf.concat(attns, axis=-1)
#             a_new = tf.reshape(tf.transpose(h_1), [1, 8, hid_units[0], dim])
#             pooling = tf.nn.max_pool(a_new, [1, 8, 1, 1], [1, 1, 1, 1], padding='VALID')
#             temp = tf.reshape(pooling, [hid_units[0], dim])
#             embed = tf.transpose(temp)
#
#             # for i in range(1, len(hid_units)):
#             #     h_old = h_1
#             #     attns = []
#             #     for _ in range(n_heads[i]):
#             #         attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
#             #                                       out_sz=hid_units[i],
#             #                                       activation=activation,
#             #                                       in_drop=ffd_drop,
#             #                                       coef_drop=attn_drop, residual=residual))
#             #
#             #     h_1 = tf.concat(attns, axis=-1)
#             embed_list.append(tf.expand_dims(tf.squeeze(embed), axis=1))
#
#         multi_embed = tf.concat(embed_list, axis=1)
#         final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
#                                                      time_major=False,
#                                                      return_alphas=True)
#
#
#         return  final_embed
class HeteGAT():
    def __init__(self,placeholders,input_dim_user,input_dim_item, user_dim, item_dim, learning_rate,inputs_list1,inputs_list2, nb_nodes1,nb_nodes2, training, attn_drop, ffd_drop,
                  bias_mat_list1,bias_mat_list2, dij_user,dij_item,hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=128):
        self.lr=0.005
        self.l2_coef=0.001
        self.placeholders = placeholders
        self.negative = placeholders['negative']
        self.length = user_dim
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.inputs_list1=inputs_list1
        self.inputs_list2=inputs_list2
        self.user = HeteGAT_multi.inference(self,inputs_list1, nb_nodes1, training, attn_drop, ffd_drop,
                  bias_mat_list1,dij_user,hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=16)
        self.item = HeteGAT_multi.inference(self,inputs_list2, nb_nodes2, training, attn_drop, ffd_drop,
                  bias_mat_list2,dij_item, hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=16)
        self.layers = []
        self.rate_matrix = None
        self.result = None
        self.los = 0
        self.hrat1 = 0
        self.hrat5 = 0
        self.hrat10 = 0
        self.hrat20 = 0
        self.ndcg5 = 0
        self.ndcg10 = 0
        self.ndcg20 = 0
        self.mrr = 0
        self.err = None
        self.auc = 0
        # self.mse = 0
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = None

        self.build()

    def build(self):
        self.layers.append(RateLayer(self.user, self.item, user_dim=self.user_dim,item_dim=self.item_dim))
        output = None
        for i in range(len(self.layers)):
            output = self.layers[i]()
        self.rate_matrix = output
        # topk = tf.nn.top_k(test, k).indices
        self.loss()

        self.train()
        self.env()

    def train(self):
        self.train_op = self.optimizer.minimize(self.los)

    def env(self):
        self.result = tf.nn.top_k(self.rate_matrix, 10).indices
        self.hrat()
        self.ndcg()
        self.mr()
        self.au()
        # self.ms()

    def loss(self):
        rating_matrix = self.placeholders['rating']
        for i in range(len(self.layers)):
            for var in self.layers[i].vars.values():
                self.los += FLAGS.weight_decay * tf.nn.l2_loss(var)
        self.los += tf.losses.mean_squared_error(rating_matrix, self.rate_matrix)

        # self.los += tf.reduce_mean(rating_matrix*tf.log(rating_matrix)-rating_matrix*tf.log(self.rate_matrix))

    def hrat(self):
        self.hrat1 = hr(self.rate_matrix, self.negative, self.length, k=1)
        self.hrat5 = hr(self.rate_matrix, self.negative, self.length, k=5)
        self.hrat10 = hr(self.rate_matrix, self.negative, self.length, k=10)
        self.hrat20 = hr(self.rate_matrix, self.negative, self.length, k=20)

    def ndcg(self):
        self.ndcg5 = ndcg(self.rate_matrix, self.negative, self.length, k=5)
        self.ndcg10 = ndcg(self.rate_matrix, self.negative, self.length, k=10)
        self.ndcg20 = ndcg(self.rate_matrix, self.negative, self.length, k=20)

    def mr(self):
        self.mrr = mrr(self.rate_matrix, self.negative, self.length)

    def au(self):
        self.auc = auc(self.rate_matrix, self.negative, self.length)