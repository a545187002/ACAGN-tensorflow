import process
from utils import *

from models import *
import time
import tensorflow as tf
tf.reset_default_graph()
import numpy as np
import os
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
# Set
learning_rate = 0.001
decay_rate = 1
global_steps = 80000
decay_steps = 100
batch_size=1
hid_units = [8]
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('output_dim', 64, 'Output_dim of user final embedding.')
flags.DEFINE_integer('latent_dim', 30,'Latent_dim of user&item.')


# Load data

support_string_user = ['uctcu','ucu','uvu'] #UCU UVU UCTCU UKU
support_string_item = ['cuc','ckc']

rating, features_item, features_user, support_user, support_item, negative,dij_user,dij_item = load_data(user=support_string_user,
                                                                                item=support_string_item)

truefeatures_list1=[features_user,features_user,features_user]
truefeatures_list2=[features_item,features_item]
dij_user_list=np.expand_dims(dij_user,1)
dij_item_list=np.expand_dims(dij_item,1)
adj_list1=support_user
adj_list2=support_item
nb_nodes1=truefeatures_list1[0].shape[0]
ft_size1=truefeatures_list1[0].shape[1]
nb_nodes2=truefeatures_list2[0].shape[0]
ft_size2=truefeatures_list2[0].shape[1]
user_dim = rating.shape[0]
item_dim = rating.shape[1]

truefeatures_list1 = [fea[np.newaxis] for fea in truefeatures_list1]
truefeatures_list2 = [fea[np.newaxis] for fea in truefeatures_list2]


adj_list1 = [adj[np.newaxis] for adj in adj_list1]
adj_list2 = [adj[np.newaxis] for adj in adj_list2]
biases_list1 = [process.adj_to_bias(adj, [nb_nodes1], nhood=1) for adj in adj_list1]
biases_list2 = [process.adj_to_bias(adj, [nb_nodes2], nhood=1) for adj in adj_list2]
attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')
attn_drop=0.6
ffd_drop=0.6


# user_support
support_num_user = len(support_string_user)
# item_support
support_num_item = len(support_string_item)
# Define placeholders
placeholders = {
    'rating': tf.placeholder(dtype=tf.float32, shape=rating.shape, name="rating"),
    'features_user': tf.placeholder(dtype=tf.float32, shape=features_user.shape, name='features_user'),
    'features_item': tf.placeholder(dtype=tf.float32, shape=features_item.shape, name="features_item"),
    'support_user': [tf.placeholder(dtype=tf.float32, name='support'+str(_)) for _ in range(support_num_user)],
    'support_item': [tf.placeholder(dtype=tf.float32, name='support'+str(_)) for _ in range(support_num_item)],
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
    'negative': tf.placeholder(dtype=tf.int32, shape=negative.shape, name='negative')
}
global_ = tf.Variable(tf.constant(0))
learning = tf.train.exponential_decay(learning_rate,global_,decay_steps,decay_rate,staircase=False)
# Create Model
model = HeteGAT(placeholders, input_dim_user=features_user.shape[1], input_dim_item=features_item.shape[1],
                user_dim=user_dim, item_dim=item_dim, learning_rate=learning,inputs_list1=truefeatures_list1,inputs_list2=truefeatures_list2,nb_nodes1=nb_nodes1,nb_nodes2=nb_nodes2,training=is_train,attn_drop=attn_drop,ffd_drop=ffd_drop,bias_mat_list1=biases_list1
                ,bias_mat_list2=biases_list2,dij_user=dij_user_list,dij_item=dij_item_list,hid_units=hid_units,n_heads=n_heads,residual=residual,activation=nonlinearity)

# Initialize session
sess = tf.Session()

# Init variables
sess.run( tf.initialize_all_variables())

# Train model
epoch = 0
averauc = 0
averhr1 = 0
averhr5 = 0
averhr10 = 0
averhr20 = 0
avernbcg5 = 0
avernbcg10 = 0
avernbcg20 = 0
avermrr = 0
bestauc=0
while epoch < global_steps:

    # Construct feed dictionary
    feed_dict = construct_feed_dict(placeholders, features_user, features_item, rating, support_user,
                                    support_item, negative)

    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({global_:epoch})

    _, los, HR1, HR5, HR10, HR20, NDCG5, NDCG10, NDCG20, MRR, AUC, user, item, result = sess.run([model.train_op, model.los, model.hrat1,
                                                                              model.hrat5, model.hrat10, model.hrat20,
                                                                              model.ndcg5, model.ndcg10, model.ndcg20,
                                                                              model.mrr, model.auc, model.user, model.item, model.result], feed_dict)

    if (epoch >= global_steps - 50):
        averauc += AUC
        averhr1 += HR1
        averhr5 += HR5
        averhr10 += HR10
        averhr20 += HR20
        avernbcg5 += NDCG5
        avernbcg10 += NDCG10
        avernbcg20 += NDCG20
        avermrr += MRR
    if AUC > bestauc:
        bestauc = AUC
        bestHR1 = HR1
        bestHR5 = HR5
        bestHR10 = HR10
        bestHR20 = HR20
        bestNDCG5 = NDCG5
        bestNDCG10 = NDCG10
        bestNDCG20 = NDCG20
        bestMRR = MRR
    if epoch % 50 == 0:
        aLine = 'Train' + str(epoch) + " Loss:" + str(los) + " HR@1:" + str(HR1) + " HR@5:" + str(
            HR5) + " HR@10:" + str(HR10) + \
                " HR@20:" + str(HR20) + " nDCG@5:" + str(NDCG5) + " nDCG@10:" + str(NDCG10) + " nDCG@20:" + str(
            NDCG20) + \
                " MRR:" + str(MRR) + " AUC:" + str(AUC)
        print(aLine)
    if epoch == global_steps - 1:
        aLine = 'Best: ' + " HR@1:" + str(bestHR1) + " HR@5:" + str(
            bestHR5) + " HR@10:" + str(bestHR10) + \
                " HR@20:" + str(bestHR20) + " nDCG@5:" + str(bestNDCG5) + " nDCG@10:" + str(
            bestNDCG10) + " nDCG@20:" + str(
            bestNDCG20) + \
                " MRR:" + str(bestMRR) + " AUC:" + str(bestauc) + '          ' + 'Averge: ' + " HR@1:" + str(
            averhr1 / 50.0) + " HR@5:" + str(
            averhr5 / 50.0) + " HR@10:" + str(averhr10 / 50.0) + \
                " HR@20:" + str(averhr20 / 50.0) + " nDCG@5:" + str(avernbcg5 / 50.0) + " nDCG@10:" + str(
            avernbcg10 / 50.0) + " nDCG@20:" + str(
            avernbcg20 / 50.0) + \
                " MRR:" + str(avermrr / 50.0) + " AUC:" + str(averauc / 50.0)
        print(aLine)
    epoch += 1
