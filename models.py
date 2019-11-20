from layers import *
import tensorflow as tf
import numpy as np

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
        self.activations_state = []
        self.activations_influence = []
        self.n_nodes = None
        self.inputs_state = None
        self.inputs_influence = None
        self.outputs = None
        self.FLAGS = None
        self.loss = 0
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model, make the output prediction
        self.activations_state.append(self.inputs_state)
        self.activations_influence.append(self.inputs_influence)
        for layer in self.layers:
            hidden_state,hidden_influence = layer(self.activations_state[-1],self.activations_influence[-1])
            self.activations_state.append(hidden_state)
            self.activations_influence.append(hidden_influence)
        self.outputs = tf.nn.tanh(self.activations_state[-1])
        self.outputs = tf.multiply(self.outputs, 1 - self.placeholders["Xs"]) + self.placeholders["Xs"]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()

        var_list1 = [var for var in tf.trainable_variables() if not 'graph_' in var.name]
        var_list2 = [var for var in tf.trainable_variables() if 'graph_' in var.name]
        opt1 = tf.train.AdamOptimizer(learning_rate=self.FLAGS.learning_rate)
        opt2 = tf.train.AdamOptimizer(learning_rate=self.FLAGS.graph_learning_rate)
        grads = tf.gradients(self.loss, var_list1 + var_list2)
        grads1 = [tf.clip_by_norm(grad, self.FLAGS.max_grad_norm) for grad in grads[:len(var_list1)]]
        grads2 = [tf.clip_by_norm(grad, self.FLAGS.max_grad_norm) for grad in grads[len(var_list1):]]
        train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
        self.opt_op = tf.group(train_op1, train_op2)

    def _loss(self):
        raise NotImplementedError

class CoupledGNN(Model):
    def __init__(self, FLAGS,init_values,placeholders,node_input_features, n_nodes, **kwargs):
        super(CoupledGNN, self).__init__(**kwargs)
        self.FLAGS = FLAGS
        self.n_nodes = n_nodes
        self.placeholders = placeholders
        self.n_layers = FLAGS.n_layers
        self.influence_dim = len(node_input_features[0])

        self.values = tf.convert_to_tensor(init_values, dtype=tf.float32)
        self.indices = self.placeholders['support_indices']

        #initialize self activation parameters
        with tf.variable_scope(self.name):
            self.initializer_layer = tf.random_uniform_initializer(minval=0.0, maxval=0.01, dtype=tf.float32)
            self.self_activation = tf.get_variable(name='graph_self_activation',shape=[self.n_nodes, 1],initializer=self.initializer_layer)

        #get input influence of each user
        self.input_features = tf.convert_to_tensor(node_input_features, dtype=tf.float32)
        self.inputs_influence = tf.contrib.layers.instance_norm(tf.transpose(tf.tile(
                                            tf.reshape(self.input_features,[self.n_nodes, self.influence_dim,1]),
                                            multiples=[1,1,self.FLAGS.batch_size]),perm=[2,0,1])
                                        ,data_format="NHWC")
        #get input state of each user
        inputs_state = self.placeholders['Xs']+ tf.tile(
                                            tf.reshape(self.self_activation,[1,self.n_nodes,1]),
                                            multiples=[self.FLAGS.batch_size,1,1])
        self.inputs_state = tf.multiply(inputs_state, 1 - self.placeholders["Xs"]) + self.placeholders["Xs"]

        self.build()

    def _loss(self):
        # regularization of l2
        for var in tf.trainable_variables():
            self.loss += self.FLAGS.reg_l2 * tf.nn.l2_loss(var)


        #mean relative square loss
        self.popularity_pre = tf.reduce_sum(self.outputs,axis=[1,2])
        self.popularity_true = tf.reduce_sum(self.placeholders['y'],axis=[1])
        self.error = tf.reduce_mean(tf.square((self.popularity_pre-self.popularity_true)/self.popularity_true))

        #regularization of cross entropy
        self.activation_pre = tf.minimum(tf.maximum(tf.reduce_sum(self.outputs,axis=2),1e-3),1-(1e-3))
        self.cross_entropy = tf.reduce_mean(
                            -tf.multiply(self.placeholders['y'],tf.log(self.activation_pre))\
                             -tf.multiply(1-self.placeholders['y'],tf.log(1-self.activation_pre)))

        #the total loss to be optimized
        self.loss += self.error +self.FLAGS.reg_cross_entropy*self.cross_entropy


    def _build(self):

        for i in range(self.n_layers):
            self.layers.append(GraphConvolution(influence_dim=self.influence_dim,
                                            flags=self.FLAGS,
                                            n_nodes = self.n_nodes,
                                            placeholders=self.placeholders,
                                            L_values = self.values,
                                            L_indices = self.indices,
                                            self_activation = self.self_activation,
                                            dropout=True))

