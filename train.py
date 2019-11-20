from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import sys,pickle

from utils import load_data,preprocess_adj,construct_feed_dict
from models import CoupledGNN
import numpy as np

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'artificial1', 'Dataset string.')
flags.DEFINE_string('filepath','./','file directory')
flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
flags.DEFINE_float('graph_learning_rate', 0.00005, 'Initial learning rate for graph weights.')
flags.DEFINE_float('reg_cross_entropy', 0.5, 'Weight for user cross entropy regularization')
flags.DEFINE_float('reg_l2', 1e-8, 'Weight for L2 regularization.')
flags.DEFINE_integer('n_layers', 3, 'number of gcn layers')
flags.DEFINE_integer('hidden_stategate', 20, 'the number of hidden units in state gating mechanism')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('early_stopping', 20, 'Tolerance for early stopping.')
flags.DEFINE_integer('batch_size', 5, 'number of samples in each batch')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
tf.flags.DEFINE_integer("display_step", 500, "display step.")
tf.flags.DEFINE_integer("training_iters", 50*320000 + 1, "max training iters.")
tf.flags.DEFINE_integer("max_grad_norm", 100, "gradient clip.")
flags.DEFINE_bool('normalize',True,"if normalized adj")

def print_args(FLAGS):
    print("dataset:", FLAGS.dataset)
    print("learning_rate:", FLAGS.learning_rate)
    print("graph_learning_rate:",FLAGS.graph_learning_rate)
    print("Weight for L2 regularization:", FLAGS.reg_l2)
    print("Weight for user cross entropy regularization:",FLAGS.reg_cross_entropy)
    print("number of gcn layers:",FLAGS.n_layers)
    print("the number of hidden units in state gating mechanism:", FLAGS.hidden_stategate)
    print("batch_size:",FLAGS.batch_size)
    print("early_stopping:", FLAGS.early_stopping)
    print("dropout:", FLAGS.dropout)
    sys.stdout.flush()
print_args(FLAGS)


# Load data
adj, total_train_x,total_train_y,\
total_val_x,total_val_y,total_test_x,total_test_y,inputs_features = load_data(FLAGS.dataset,FLAGS.filepath)


print(type(adj),adj.shape)
print("total number of samples in train,val and test:",
      len(total_train_x),len(total_val_x),len(total_test_x))
sys.stdout.flush()

support = preprocess_adj(adj,FLAGS.normalize)
(init_indices,init_values,shape) = support

model_func = CoupledGNN

placeholders = {
    'support_indices': tf.placeholder(tf.int64, shape=(None,2)),
    'Xs': tf.placeholder(tf.float32, shape=(None,adj.shape[0],1)),
    'y': tf.placeholder(tf.float32, shape=(None,adj.shape[0])),
    'dropout': tf.placeholder_with_default(0., shape=()),
}

# Create model
model = model_func(FLAGS,init_values,placeholders, inputs_features,n_nodes = adj.shape[0])

# Initialize session
sess = tf.Session()
# Init variables
sess.run(tf.global_variables_initializer())

# Define model evaluation function
def evaluate(Xs, feed_dict_val, y, placeholders):
    t_test = time.time()
    # feed_dict_val = construct_feed_dict(Xs, support, y, placeholders)
    feed_dict_val.update({placeholders['Xs']: Xs})
    feed_dict_val.update({placeholders['y']: y})
    outs_val = sess.run([model.loss, model.error], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

#define batch data getting function
def get_batch_data(total_x,total_y,start_index,end_index,n_nodes):
    train_x = []
    train_y = []
    for i in range(start_index,end_index):
        ob_uid = total_x[i]
        pre_uid = total_y[i]
        x = np.zeros(shape=(n_nodes,1))
        y = np.zeros(shape=(n_nodes))
        for i in range(len(ob_uid)):
            (time,uid) = ob_uid[i]
            x[uid] = 1.0
        for uid in pre_uid:
            y[uid] = 1.0
        train_x.append(x)
        train_y.append(y)
    return train_x,train_y

# Train model
start_time = time.time()
saver = tf.train.Saver()
check_file = "./"+str(FLAGS.dataset)+"/coupledgnn_lr"+str(FLAGS.learning_rate)+"_glr"+str(FLAGS.graph_learning_rate)\
             +"_l2"+str(FLAGS.reg_l2)+"_layers"+str(FLAGS.n_layers)+"_reg_cross_entropy"+str(FLAGS.reg_cross_entropy)

max_i = int(len(total_train_x)/FLAGS.batch_size)
display_step = min(FLAGS.display_step, max_i)
print("number of display step:",display_step)
step = 0
train_loss = []
train_rmse = []
best_val_loss = 1000
best_test_loss = 1000
best_val_rmse = 1000
best_test_rmse = 1000
patience = FLAGS.early_stopping
feed_dict = construct_feed_dict(support, placeholders)
while step * FLAGS.batch_size < FLAGS.training_iters:
    i = step % max_i
    # Construct feed dictionary
    train_x,train_y = get_batch_data(total_train_x,total_train_y,
                        FLAGS.batch_size*i,FLAGS.batch_size*(i+1),adj.shape[0])
    feed_dict.update({placeholders['Xs']: train_x})
    feed_dict.update({placeholders['y']: train_y})
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.error], feed_dict=feed_dict)
    train_loss.append(outs[1])
    train_rmse.append(outs[2])
    sys.stdout.flush()

    if step % display_step ==0:

        #loss in the validation set
        val_loss = []
        val_rmse = []
        for i in range(int(len(total_val_x)/FLAGS.batch_size)):
            val_x,val_y = get_batch_data(total_val_x,total_val_y,
                        FLAGS.batch_size*i,FLAGS.batch_size*(i+1),adj.shape[0])
            loss,rmse, duration = evaluate(val_x, feed_dict, val_y, placeholders)
            val_loss.append(loss)
            val_rmse.append(rmse)

        # loss in the test set
        test_loss = []
        test_rmse = []
        for i in range(int(len(total_test_x) / FLAGS.batch_size)):
            test_x, test_y = get_batch_data(total_test_x, total_test_y,
                                        FLAGS.batch_size * i, FLAGS.batch_size * (i + 1), adj.shape[0])
            loss, rmse, test_duration = evaluate(test_x, feed_dict, test_y, placeholders)
            test_loss.append(loss)
            test_rmse.append(rmse)

        #record the best result on validation set
        if np.mean(val_rmse) < best_val_rmse:
            best_val_loss = np.mean(val_loss)
            best_test_loss = np.mean(test_loss)
            best_val_rmse = np.mean(val_rmse)
            best_test_rmse = np.mean(test_rmse)
            patience = FLAGS.early_stopping
            saver.save(sess,check_file)
        print("#" + str(int(step/display_step)) +
              "\tTraining Loss= " + "{:.6f}".format(np.mean(train_loss)) +
              ", Validation Loss= " + "{:.6f}".format(np.mean(val_loss)) +
              ", Test Loss= " + "{:.6f}".format(np.mean(test_loss)) +
              "\n\tTraining rmse= " + "{:.6f}".format(np.mean(train_rmse))+
              ", Validation rmse= " + "{:.6f}".format(np.mean(val_rmse)) +
              ", Test rmse= " + "{:.6f}".format(np.mean(test_rmse)) +
              "\nBest Valid rmse= " + "{:.6f}".format(best_val_rmse) +
              ", Best Test rmse= " + "{:.6f}".format(best_test_rmse)+
              ", Patience="+str(patience)+
              ", Time=" + str(time.time()-start_time)
             )

        sys.stdout.flush()

        train_loss = []
        train_rmse = []
        patience -= 1
        #early stopping
        if not patience:
            break
    step +=1

print("Optimization Finished!")
print("Finished!\n----------------------------------------------------------------")
print("Time:", time.time()-start_time)
print("Valid Loss:", best_val_loss)
print("Valid Rmse:", best_val_rmse)
print("Test Loss:", best_test_loss)
print("Test Rmse:", best_test_rmse)
sys.stdout.flush()
