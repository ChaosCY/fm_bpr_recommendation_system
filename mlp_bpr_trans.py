import tensorflow as tf
import numpy as np

class MLP_BPR_T():
	'''
	SVD with BPR loss
	'''
	def __init__(self, args):
		self.args = args
		
		self._items = tf.placeholder(tf.int32, shape=[None, ], name='items')
		self._users_pos = tf.placeholder(tf.int32, shape=[None, ], name='users_pos')
		self._users_neg = tf.placeholder(tf.int32, shape=[None, ], name='users_neg')
		self._keep_prob = tf.placeholder(tf.float32)
				
		self._pos_fea = tf.placeholder(tf.float32, shape=[None, 29], name='pos_fea')
		self._neg_fea = tf.placeholder(tf.float32, shape=[None, 29], name='neg_fea')
		
		num_users = args.num_users+1
		num_items = args.num_items+1
		num_factors = args.num_factors
		
		with tf.variable_scope('user'):
			user_embeddings = tf.get_variable(
				name='embedding',
				shape=[num_users, num_factors],
				initializer=tf.contrib.layers.xavier_initializer(),
				regularizer=tf.contrib.layers.l2_regularizer(args.reg_lambda))
			user_bias = tf.get_variable(
				name='bias',
				shape=[num_users, ],
				initializer=tf.contrib.layers.xavier_initializer(),
				regularizer=tf.contrib.layers.l2_regularizer(args.reg_lambda))
			self.p_u_pos = tf.nn.embedding_lookup(
				user_embeddings,
				self._users_pos,
				name='p_u_pos')
			b_u_pos = tf.nn.embedding_lookup(
				user_bias,
				self._users_pos,
				name='b_u_pos')
			p_u_neg = tf.nn.embedding_lookup(
				user_embeddings,
				self._users_neg,
				name='p_u_neg')
			b_u_neg = tf.nn.embedding_lookup(
				user_bias,
				self._users_neg,
				name='b_u_neg')
		
		with tf.variable_scope('item'):
			item_embeddings = tf.get_variable(
				name='embedding',
				shape=[num_items, num_factors],
				initializer=tf.contrib.layers.xavier_initializer(),
				regularizer=tf.contrib.layers.l2_regularizer(args.reg_lambda))
			item_bias = tf.get_variable(
				name='bias',
				shape=[num_items, ],
				initializer=tf.contrib.layers.xavier_initializer(),
				regularizer=tf.contrib.layers.l2_regularizer(args.reg_lambda))
			q_i = tf.nn.embedding_lookup(
				item_embeddings,
				self._items,
				name='q_i')
			b_i = tf.nn.embedding_lookup(
				item_bias,
				self._items,
				name='b_i')
		
		with tf.variable_scope('weight'):
			hidden_w = tf.get_variable("hidden_w", [29, 10], initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(args.reg_lambda))
			hidden_b = tf.get_variable("hidden_b", [10], initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(args.reg_lambda))
			hidden_w1 = tf.get_variable("hidden_w1", [10,10], initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(args.reg_lambda))
			hidden_b1 = tf.get_variable("hidden_b1", [10], initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(args.reg_lambda))
		
		pred_pos = tf.multiply(self.p_u_pos, q_i)
		pred_pos = pred_pos + 1e-5
		hidden = tf.nn.relu(tf.matmul(self._pos_fea, hidden_w) + hidden_b)
		hidden1 = tf.nn.sigmoid(tf.matmul(hidden, hidden_w1) + hidden_b1)
		pred_pos = tf.reduce_sum(tf.multiply(pred_pos, hidden1), axis=1)
		self._pred = tf.add_n([b_u_pos, b_i, pred_pos])
		
		pred_neg = tf.multiply(p_u_neg, q_i)
		pred_neg = pred_neg + 1e-5
		hidden = tf.nn.relu(tf.matmul(self._neg_fea, hidden_w) + hidden_b)
		hidden1 = tf.nn.sigmoid(tf.matmul(hidden, hidden_w1) + hidden_b1)
		pred_neg = tf.reduce_sum(tf.multiply(pred_neg, hidden1), axis=1)
		self._pred_neg = tf.add_n([b_u_neg, b_i, pred_neg])
		
		x = tf.subtract(self._pred, self._pred_neg)
		self.loss = -tf.reduce_sum(tf.log(tf.sigmoid(x)))
		self._loss = tf.add(self.loss, tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), name='objective')
		
		self.lr = tf.Variable(0.0, trainable=False)
		self._optimizer = tf.train.AdamOptimizer(self.lr).minimize(self._loss, name='optimizer')
