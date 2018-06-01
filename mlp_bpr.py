import tensorflow as tf
import numpy as np

class MLP_BPR():
	'''
	a MLP model with BPR loss
	'''
	def __init__(self, args):
		self.args = args
		
		self._users = tf.placeholder(tf.int32, shape=[None, ], name='users')
		self._items_pos = tf.placeholder(tf.int32, shape=[None, ], name='items_pos')
		self._items_neg = tf.placeholder(tf.int32, shape=[None, ], name='items_neg')
		
		self._pos_fea = tf.placeholder(tf.float32, shape=[None, 16], name='pos_fea')
		self._neg_fea = tf.placeholder(tf.float32, shape=[None, 16], name='neg_fea')
		
		num_users = args.num_users+1
		num_items = args.num_items+1
		num_factors = args.num_factors
		
		with tf.variable_scope('user'):
			user_embeddings = tf.get_variable(
				name='embedding',
				shape=[num_users, num_factors],
				initializer=tf.random_normal_initializer(),
				regularizer=tf.contrib.layers.l2_regularizer(args.reg_lambda))
			user_bias = tf.get_variable(
				name='bias',
				shape=[num_users, ],
				initializer=tf.random_normal_initializer(),
				regularizer=tf.contrib.layers.l2_regularizer(args.reg_lambda))
			self.p_u = tf.nn.embedding_lookup(
				user_embeddings,
				self._users,
				name='p_u')
			self.b_u = tf.nn.embedding_lookup(
				user_bias,
				self._users,
				name='b_u')
		
		with tf.variable_scope('item'):
			item_embeddings = tf.get_variable(
				name='embedding',
				shape=[num_items, num_factors],
				initializer=tf.random_normal_initializer(),
				regularizer=tf.contrib.layers.l2_regularizer(args.reg_lambda))
			item_bias = tf.get_variable(
				name='bias',
				shape=[num_items, ],
				initializer=tf.random_normal_initializer(),
				regularizer=tf.contrib.layers.l2_regularizer(args.reg_lambda))
			self.q_i_pos = tf.nn.embedding_lookup(
				item_embeddings,
				self._items_pos,
				name='q_i_pos')
			self.b_i_pos = tf.nn.embedding_lookup(
				item_bias,
				self._items_pos,
				name='b_i_pos')
			q_i_neg = tf.nn.embedding_lookup(
				item_embeddings,
				self._items_neg,
				name='q_i_neg')
			b_i_neg = tf.nn.embedding_lookup(
				item_bias,
				self._items_neg,
				name='b_i_neg')
		
		with tf.variable_scope('weight'):
			hidden_w = tf.get_variable("hidden_w", [num_factors, 10], initializer=tf.random_normal_initializer())
			hidden_b = tf.get_variable("hidden_b", [10], initializer=tf.random_normal_initializer())
			hidden_w1 = tf.get_variable("hidden_w1", [16, 5], initializer=tf.random_normal_initializer())
			hidden_b1 = tf.get_variable("hidden_b1", [5], initializer=tf.random_normal_initializer())
			out_w = tf.get_variable("out_w", [15, 1], initializer=tf.random_normal_initializer())
			out_b = tf.get_variable("out_b", [1], initializer=tf.random_normal_initializer())
		
		emb_concat_pos = tf.multiply(self.p_u, self.q_i_pos)
		hidden = tf.nn.relu(tf.matmul(emb_concat_pos, hidden_w) + hidden_b)
		hidden1 = tf.nn.relu(tf.matmul(self._pos_fea, hidden_w1) + hidden_b1)
		hidden = tf.concat([hidden, hidden1], 1)
		out = tf.nn.sigmoid(tf.matmul(hidden, out_w) + out_b)
		out = tf.reshape(out, [-1])
		self._pred = tf.add_n([self.b_u, self.b_i_pos, out])
		
		emb_concat_neg = tf.multiply(self.p_u, q_i_neg)
		hidden = tf.nn.relu(tf.matmul(emb_concat_neg, hidden_w) + hidden_b)
		hidden1 = tf.nn.relu(tf.matmul(self._neg_fea, hidden_w1) + hidden_b1)
		hidden = tf.concat([hidden, hidden1], 1)
		out = tf.nn.sigmoid(tf.matmul(hidden, out_w) + out_b)
		out = tf.reshape(out, [-1])
		self._pred_neg = tf.add_n([self.b_u, b_i_neg, out])
		
		x = tf.subtract(self._pred, self._pred_neg)
		self.loss = -tf.reduce_sum(tf.log(tf.sigmoid(x)))
		self._loss = tf.add(self.loss, tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), name='objective')
		
		self.lr = tf.Variable(0.0, trainable=False)
		self._optimizer = tf.train.AdamOptimizer(self.lr).minimize(self._loss, name='optimizer')
