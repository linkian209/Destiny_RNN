import tensorflow as tf
import numpy as np

def GRUTensor(hidden_dim=128, vocab_size=256, batch_size=32, num_layers=3,
              learning_rate=0.0001, num_steps=2000, decay=0.9):

    # Reset default graph
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

    # Generate inital stuff
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels')
    dropout = tf.constant(1.0)
    embeddings = tf.get_variable('embedding_matrix',
                 [vocab_size, hidden_dim])

    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    # Assemble the Network
    cell = tf.nn.rnn_cell.GRUCell(hidden_dim)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)

    # Initial State
    init_state = cell.zero_state(batch_size, tf.float32)

    # Compile Network
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    # Inner Workings
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [hidden_dim, vocab_size])
        b = tf.get_variable('b', [vocab_size], initializer=tf.constant_initializer(0.0))

    # Reshape Outputs
    rnn_outputs = tf.reshape(rnn_outputs, [-1, hidden_dim])
    y_reshaped = tf.reshape(y, [-1])

    # Calculations
    logits = tf.matmul(rnn_outputs, W) + b
    predictions = tf.nn.softmax(logits)
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))
    train_step = tf.train.RMSPropOptimizer(learning_rate, decay=decay).minimize(total_loss)

    # Return model wrapper
    return dict(x=x, y=y, init_state=init_state, final_state=final_state,
                total_loss=total_loss, train_step=train_step, preds=predictions,
                saver=tf.train.Saver())
