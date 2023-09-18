import numpy as np
import tensorflow as tf

from utils import structural_interaction
flags = tf.app.flags
FLAGS = flags.FLAGS
conv1d = tf.layers.conv1d




def attn_head(seq, out_sz, bias_mat,activation,adj,in_drop=0.0, coef_drop=0.0, residual=False,
              return_coef=False):
    """[summary]

    [description]

    Arguments:
        seq {[type]} -- shape=(batch_size, nb_nodes, fea_size))

    """
    # seq 输入的节点特征矩阵 numgraph，num_node，fea_size
    # out_sz 变换后的节点特征维度 Whi后的节点表示维度
    # bias_mat 经过变换后的邻接矩阵，大小为 num_node,num_node

    with tf.name_scope('my_attn'):
        W_s = tf.Variable(tf.random_normal([1,1]))
        W_e = tf.Variable(tf.random_normal([1,1]))
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        seq_fts = tf.layers.conv1d(seq, out_sz, 1,
                                   use_bias=False)  # 对seq利用卷积核大小为1的1D卷积投影变换得到seq_fts，维度为out_sz。shape:[num_graph,num_code,out_sz]

        f_1 = tf.layers.conv1d(seq_fts, 1, 1)  # 继续卷积 得到节点本身的投影f_1和其邻居的投影f_2 [num_graph,num_code,1]
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)

        logits = f_1 + tf.transpose(f_2, [0, 2, 1])  # f_2转置与f_1叠加 得到[num_graph,num_node,num_node]的logits，注意力矩阵
        e = tf.nn.leaky_relu(logits)   # 对logits进行softmax归一化得到注意力权重
        s=tf.cast(adj,tf.float32)
        a = abs(W_s) * s + abs(W_e) * e
        att=tf.nn.softmax(a+bias_mat)
        if coef_drop != 0.0:
            att = tf.nn.dropout(att, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(att, seq_fts)  # 注意力矩阵乘特征矩阵，得到更新后的节点表示vals
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                seq_fts = ret + seq
        if return_coef:
            return activation(ret), att
        else:
            return activation(ret)  # activation
def attn_head1(seq, out_sz, bias_mat,activation, in_drop=0.0, coef_drop=0.0, residual=False,
              return_coef=False):
    """[summary]

    [description]

    Arguments:
        seq {[type]} -- shape=(batch_size, nb_nodes, fea_size))

    """
    # seq 输入的节点特征矩阵 numgraph，num_node，fea_size
    # out_sz 变换后的节点特征维度 Whi后的节点表示维度
    # bias_mat 经过变换后的邻接矩阵，大小为 num_node,num_node

    with tf.name_scope('my_attn'):

        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        seq_fts = tf.layers.conv1d(seq, out_sz, 1,
                                   use_bias=False)  # 对seq利用卷积核大小为1的1D卷积投影变换得到seq_fts，维度为out_sz。shape:[num_graph,num_code,out_sz]

        f_1 = tf.layers.conv1d(seq_fts, 1, 1)  # 继续卷积 得到节点本身的投影f_1和其邻居的投影f_2 [num_graph,num_code,1]
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)

        logits = f_1 + tf.transpose(f_2, [0, 2, 1])  # f_2转置与f_1叠加 得到[num_graph,num_node,num_node]的logits，注意力矩阵
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)  # 对logits进行softmax归一化得到注意力权重

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)  # 注意力矩阵乘特征矩阵，得到更新后的节点表示vals
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                seq_fts = ret + seq
        if return_coef:
            return activation(ret), coefs
        else:
            return activation(ret)  # activation
def attn_head2(seq, out_sz, bias_mat,activation,adj,in_drop=0.0, coef_drop=0.0, residual=False,
              return_coef=False):
    """[summary]

    [description]

    Arguments:
        seq {[type]} -- shape=(batch_size, nb_nodes, fea_size))

    """
    # seq 输入的节点特征矩阵 numgraph，num_node，fea_size
    # out_sz 变换后的节点特征维度 Whi后的节点表示维度
    # bias_mat 经过变换后的邻接矩阵，大小为 num_node,num_node

    with tf.name_scope('my_attn'):
        W_s = tf.Variable(tf.random_normal([1,1]))
        W_e = tf.Variable(tf.random_normal([1,1]))
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        seq_fts = tf.layers.conv1d(seq, out_sz, 1,
                                   use_bias=False)  # 对seq利用卷积核大小为1的1D卷积投影变换得到seq_fts，维度为out_sz。shape:[num_graph,num_code,out_sz]

        f_1 = tf.layers.conv1d(seq_fts, 1, 1)  # 继续卷积 得到节点本身的投影f_1和其邻居的投影f_2 [num_graph,num_code,1]
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)

        logits = f_1 + tf.transpose(f_2, [0, 2, 1])  # f_2转置与f_1叠加 得到[num_graph,num_node,num_node]的logits，注意力矩阵
        e = tf.nn.leaky_relu(logits)   # 对logits进行softmax归一化得到注意力权重
        s=tf.cast(adj,tf.float32)
        a = abs(W_s) * s + abs(W_e) * e
        att=tf.nn.softmax(s+bias_mat)
        if coef_drop != 0.0:
            att = tf.nn.dropout(att, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(att, seq_fts)  # 注意力矩阵乘特征矩阵，得到更新后的节点表示vals
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                seq_fts = ret + seq
        if return_coef:
            return activation(ret), att
        else:
            return activation(ret)
class RatLayer():
    def __init__(self, user, item, act=tf.nn.relu):
        self.user = user
        self.item = item
        self.act = act

    def __call__(self):
        rate_matrix = tf.matmul(self.user,tf.transpose(self.item))
        return self.act(rate_matrix)


class RateLayer():
    def __init__(self, user, item, user_dim, item_dim, ac=tf.nn.relu):
        self.user = user
        self.item = item
        self.name = 'RateLayer'
        self.ac = ac
        self.vars = {}
        with tf.name_scope(self.name + '_vars'):
            self.vars['user_latent'] = tf.Variable(tf.truncated_normal(shape=[int(FLAGS.latent_dim), user_dim],
                                                                       stddev=1.0), name='user_latent_matrix')
            self.vars['item_latent'] = tf.Variable(tf.truncated_normal(shape=[int(FLAGS.latent_dim), item_dim],
                                                                       stddev=1.0), name='item_latent_matrix')
            self.vars['user_specific'] = tf.Variable(tf.truncated_normal(shape=[int(FLAGS.output_dim), item_dim],
                                                                         stddev=0.1), name='user_specific')
            self.vars['item_specific'] = tf.Variable(tf.truncated_normal(shape=[int(FLAGS.output_dim), user_dim],
                                                                         stddev=0.1), name='item_specific')
            self.vars['user_bias'] = tf.zeros(shape=[user_dim,1],name='user_bias')
            self.vars['item_bias'] = tf.zeros(shape=[item_dim,1], name='item_bias')
            self.vars['alpha1'] = tf.Variable(initial_value=1.0, name='alpha1')
            self.vars['alpha2'] = tf.Variable(initial_value=1.0, name='alpha2')

    def __call__(self):
        rate_matrix1 = tf.matmul(tf.transpose(self.vars['user_latent']),self.vars['item_latent'])
        u_matrix = self.vars['alpha1']*(tf.matmul(self.user, self.vars['user_specific'])+self.vars['user_bias'])
        i_matrix = self.vars['alpha2']*(tf.transpose(tf.matmul(self.item,
                                                               self.vars['item_specific'])+self.vars['item_bias']))
        rate_matrix2 = rate_matrix1+u_matrix+i_matrix
        return rate_matrix2
def SimpleAttLayer(inputs, attention_size, time_major=False, return_alphas=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas