import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def dot(x, y, input_dim,output_dim,n_nodes,sparse=False,a_is_sparse = False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        y = tf.reshape(tf.transpose(y,perm=[1,0,2]),[n_nodes,-1])

        res = tf.sparse_tensor_dense_matmul(x, y)
        res = tf.transpose(tf.reshape(res,[n_nodes,-1,output_dim]),perm=[1,0,2])
    else:
        x = tf.reshape(x,[-1,input_dim])
        res = tf.matmul(x, y,a_is_sparse=a_is_sparse)
        res = tf.reshape(res,[-1,n_nodes,output_dim])
    return res


# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(object):
    """Base layer class.
    # Properties
        name: String, defines the variable scope of the layer.
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging

    def _call(self, inputs_state,inputs_influence):
        return inputs_state,inputs_influence

    def __call__(self, inputs_state,inputs_influence):
        with tf.name_scope(self.name):
            outputs_state,outputs_influence = self._call(inputs_state,inputs_influence)
            return outputs_state,outputs_influence


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, influence_dim,flags, n_nodes,placeholders,
                 L_values,L_indices,self_activation,dropout=0., **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.L_values = L_values
        self.L_indices = L_indices
        self.self_activation = self_activation

        self.influence_dim = influence_dim
        self.n_nodes = n_nodes
        self.batch_size = flags.batch_size

        self.placeholders = placeholders
        tf.set_random_seed(-1)
        # helper variable for sparse dropout
        self.initializer_layer = tf.random_uniform_initializer(minval=-0.01, maxval=0.01, dtype=tf.float32)

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weight_trans_'] = tf.get_variable(name='weight_trans_' ,
                                                                  shape=[self.influence_dim,self.influence_dim],
                                                initializer=self.initializer_layer)
            #parameters in state graph neural networks
            self.vars['state_beta' ] = tf.get_variable(name='state_beta',shape=[2*self.influence_dim,1],
                                                 initializer= self.initializer_layer)
            self.vars['state_weight_self' ] = tf.get_variable(name='state_weight_self' ,shape=(),
                                                initializer=self.initializer_layer)
            self.vars['state_weight_neighbor' ] = tf.get_variable(name='state_weight_neighbor' ,shape=(),
                                                initializer=self.initializer_layer)

            # parameters in influence graph neural networks
            self.n_step = flags.hidden_stategate
            self.vars['stategating_weight1' ] = tf.get_variable('stategating_weight1',shape=[1, self.n_step],
                                                initializer=self.initializer_layer)

            self.vars['stategating_weight2' ] = tf.get_variable('stategating_weight2',shape=[self.n_step, 1],
                                                initializer=self.initializer_layer)
            self.vars['stategating_biase1' ] = tf.get_variable('stategating_biase1',shape=[1,self.n_step],
                                                initializer=self.initializer_layer)

            self.vars['stategating_biase2' ] = tf.get_variable('stategating_biase2',shape=[1,1],
                                                initializer=self.initializer_layer)
            self.vars['influence_attention' ] = tf.get_variable(name='influence_attention',shape=[2*self.influence_dim,1],
                                                initializer= self.initializer_layer)

            self.vars['influence_weight_self' ] = tf.get_variable(name='influence_weight_self' ,shape=(),
                                                initializer=self.initializer_layer)
            self.vars['influence_weight_neighbor' ] = tf.get_variable(name='influence_weight_neighbor',shape=(),
                                                initializer=self.initializer_layer)

    def _call(self, inputs_state,inputs_influence):
        x_state = inputs_state
        x_influence = inputs_influence

        self.support_gcn = tf.SparseTensorValue(
                indices=self.L_indices,values=self.L_values,dense_shape=(self.n_nodes,self.n_nodes))
        [L_indices_row, L_indices_col] = tf.split(self.L_indices, num_or_size_splits=2, axis=1)
        batch_size = self.batch_size


        #feature transformation for influence representation
        transformed_feature = dot(x_influence, self.vars['weight_trans_' ],
                          self.influence_dim, self.influence_dim, self.n_nodes,
                          sparse=False, a_is_sparse=False)
        split_x_transformed_feature = tf.split(transformed_feature, num_or_size_splits=batch_size, axis=0)


        #------------------layer in stage graph neural networks------------------------
        support_state_batch = []
        split_filtered_features_s = tf.split(x_state, num_or_size_splits=batch_size, axis=0)

        for j in range(batch_size):
            each_split_x_transformed_feature = tf.reshape(split_x_transformed_feature[j],
                                                              [self.n_nodes, self.influence_dim])
            #get corresponding influence representation for influence gate function
            L_indices_row_repre = tf.nn.embedding_lookup(each_split_x_transformed_feature, tf.reduce_sum(L_indices_row, axis=1))
            L_indices_col_repre = tf.nn.embedding_lookup(each_split_x_transformed_feature, tf.reduce_sum(L_indices_col, axis=1))

            # calculate influence gate function
            L_attention_value_s = tf.reduce_sum(tf.nn.leaky_relu(
                        tf.matmul(tf.concat([L_indices_row_repre, L_indices_col_repre], axis=1),
                                  self.vars['state_beta' ]), alpha=0.02), axis=1)
            self.support_s = tf.SparseTensorValue(
                        indices=self.L_indices, values=L_attention_value_s, dense_shape=(self.n_nodes, self.n_nodes))

            # update for state
            S_neighbor_info = dot(self.support_s, split_filtered_features_s[j], 1, 1, self.n_nodes, sparse=True)+self.self_activation

            S_update = tf.nn.elu(
                    self.vars['state_weight_self' ] * split_filtered_features_s[j]
                    + self.vars['state_weight_neighbor' ] * S_neighbor_info)

            support_state_batch.append(S_update)

        output_state_ = tf.concat(support_state_batch, axis=0)
        output_state = tf.multiply(output_state_,1-self.placeholders["Xs"])+self.placeholders["Xs"]

        # ------------------layer in influence graph neural networks------------------------
        #state gatting for influence representation
        x_state_e = tf.reshape(tf.nn.elu(tf.matmul(
                                    tf.nn.elu(tf.matmul(tf.reshape(x_state,[-1,1]),
                                    self.vars['stategating_weight1' ])+self.vars['stategating_biase1' ]),
                            self.vars['stategating_weight2' ])+self.vars['stategating_biase2' ]),
                            [-1,self.n_nodes,1])

        support_influence_batch = []
        filtered_features_e = tf.multiply(transformed_feature,
                                    tf.tile(x_state_e,multiples=[1,1,self.influence_dim]))
        split_filtered_features_e = tf.split(filtered_features_e, num_or_size_splits=batch_size, axis=0)
        for j in range(batch_size):
            each_split_x_transformed_feature = tf.reshape(split_x_transformed_feature[j],
                                                              [self.n_nodes, self.influence_dim])


            #get influence representation for attention mechanism
            L_indices_row_repre = tf.nn.embedding_lookup(each_split_x_transformed_feature, tf.reduce_sum(L_indices_row, axis=1))
            L_indices_col_repre = tf.nn.embedding_lookup(each_split_x_transformed_feature, tf.reduce_sum(L_indices_col, axis=1))

            #calculate attention weight
            L_attention_value_e = tf.reduce_sum(tf.nn.leaky_relu(
                        tf.matmul(tf.concat([L_indices_row_repre, L_indices_col_repre], axis=1),
                                  self.vars['influence_attention']), alpha=0.2), axis=1)
            self.support_e = tf.SparseTensorValue(
                        indices=self.L_indices, values=L_attention_value_e,
                        dense_shape=(self.n_nodes, self.n_nodes))
            self.support_e = tf.sparse_softmax(self.support_e)


            #update influence representation
            E_neighbor_info = dot(self.support_e, split_filtered_features_e[j], self.influence_dim,
                                                self.influence_dim, self.n_nodes, sparse=True)
            E_update = tf.nn.elu(self.vars['influence_weight_self' ]* split_x_transformed_feature[j]
                                    +self.vars['influence_weight_neighbor' ]*E_neighbor_info)
            support_influence_batch.append(E_update)

        output_influence = tf.concat(support_influence_batch, axis=0)


        return output_state,output_influence
