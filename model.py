import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import codecs
import random
import os
from tensorflow.python import debug as tf_debug

import utils
import mlp_bpr
import mlp_bpr_trans
import evaluation


class Model(object):
	''' train the model and make prediction through the model
	'''
	def __init__(self, args, mode, batch_loader=None):
		self.args = args
		self.batch_loader = batch_loader
		self.mode = mode
		# MLP-BPR model
		if mode=='u_p':
			self.model = mlp_bpr.MLP_BPR(args)
		elif mode=='p_u':
			self.model = mlp_bpr_trans.MLP_BPR_T(args)
	
	def train_val(self, test_X, user_pro_dict, pro_user_dict, user_df, item_df):
		''' train for validation
		test_X: test data
		user_pro_dict: user and product mapping
		pro_user_dict: product and user mapping
		user_df, item_df: user and item features
		''' 
		saver = tf.train.Saver()
		with tf.Session() as sess:
			with tf.device("/" + str(self.args.dev)):
				summaries = tf.summary.merge_all()
				writer = tf.summary.FileWriter(os.path.join(self.args.log_dir))
				writer.add_graph(sess.graph)
				
				sess.run(tf.global_variables_initializer())
				
				# remove the positive user that appears in the train
				pro_user_test = {}
				
				pred_prods = list(test_X['pid'].unique())
				users = list(user_pro_dict.keys())
				for prod in pred_prods:
					users_copy = users.copy()
					pro_user_test[prod] = self.difference_list(users_copy, pro_user_dict[prod])
				# iterate epoches
				for epoch in range(self.args.epoches):
					self.iter_epoches(sess, epoch, user_pro_dict, pro_user_dict, user_df, item_df)
					if (epoch+1)%self.args.pIter==0:
						pred_dict = {}
						pred_prods = list(test_X['pid'].unique())
						for prod in pred_prods:
							user_list = pro_user_test[prod]
							pro_list = [prod]*len(user_list)
							user_item_df = self.get_feed_data(user_list, pro_list, user_df, item_df)
							
							feed = {self.model._users_pos: user_list, self.model._items: pro_list, self.model._pos_fea: user_item_df.iloc[:,2:], self.model._keep_prob: 1}
							
							pred = sess.run(self.model._pred, feed)
							pred_list = list(pred)
							# sort by predict score
							preds_sort, users_sort, pros_sort = self.sort_predictions(pred_list, user_list, pro_list)
							pred_dict[str(prod)] = users_sort[:self.args.itu_k]
							print(str(prod) + "\t" + str(self.cal_pre_k(pred_dict, test_X[test_X['pid']==prod])))
				# save model
				save_path = saver.save(sess, "save_val/model.ckpt")
	
	def train(self, user_pro_dict, pro_user_dict, user_df, item_df):
		''' train
		user_pro_dict: user and product mapping
		pro_user_dict: product and user mapping
		user_df, item_df: user and item features
		''' 
		saver = tf.train.Saver()
		with tf.Session() as sess:
			with tf.device("/" + str(self.args.dev)):
				summaries = tf.summary.merge_all()
				writer = tf.summary.FileWriter(os.path.join(self.args.log_dir))
				writer.add_graph(sess.graph)
				
				sess.run(tf.global_variables_initializer())
				
				# iterate epoches
				for epoch in range(self.args.epoches):
					self.iter_epoches(sess, epoch, user_pro_dict, pro_user_dict, user_df, item_df)
				# save model
				save_path = saver.save(sess, "save/model.ckpt")
				
	def iter_epoches(self, sess, epoch, user_pro_dict, pro_user_dict, user_df, item_df):
		''' iterate epoches
		sess: current session
		epoch: current epoch
		user_pro_dict: user and product mapping
		pro_user_dict: product and user mapping
		user_df, item_df: user and item features
		''' 
		# learning rate decay
		sess.run(tf.assign(self.model.lr, self.args.learning_rate * (self.args.decay_rate ** epoch)))
		# reset the batch pointer to 0
		self.batch_loader.reset_batch_pointer()
		# iterate every batches
		out_list = []
		for iteration in range(self.batch_loader.num_batches):
			train_X, _ = self.batch_loader.next_batch()
			train_X['eval'] = 1
			# get the features of positive samples
			train_X_fea = utils.feature_map(train_X, user_df, item_df)
			if self.mode=='u_p':
				# u_p: users to items
				train_neg_X = utils.negative_sampling(train_X, user_pro_dict, pro_user_dict, 1, 'u_p')
				# get the features of negative samples
				train_neg_X_fea = utils.feature_map(train_neg_X, user_df, item_df)
				feed = {self.model._users: train_X['uid'], self.model._items_pos: train_X['pid'], self.model._items_neg: train_neg_X['pid'], \
						self.model._pos_fea: train_X_fea.iloc[:, 3:], \
						self.model._neg_fea: train_neg_X_fea.iloc[:, 3:]}
			elif self.mode=='p_u':
				# p_u: items to users
				train_neg_X = utils.negative_sampling(train_X, user_pro_dict, pro_user_dict, 1, 'p_u')
				train_neg_X_fea = utils.feature_map(train_neg_X, user_df, item_df)
				
				feed = {self.model._items: train_X['pid'], self.model._users_pos: train_X['uid'], self.model._users_neg: train_neg_X['uid'], \
						self.model._pos_fea: train_X_fea.iloc[:, 3:], self.model._neg_fea: train_neg_X_fea.iloc[:, 3:], self.model._keep_prob: self.args.keep_prob}
			
			pred, _, loss = sess.run([self.model._pred, self.model._optimizer, self.model._loss], feed)
			print("epoches: %3d, train loss: %2.6f" % (epoch, loss))
	
	def test(self, test_X, user_pro_dict, pro_user_dict, user_df, item_df):
		''' predict
		test_X: test data
		user_pro_dict: user and product mapping
		pro_user_dict: product and user mapping
		user_df, item_df: user and item features
		'''
		out_list = []
		saver = tf.train.Saver()
		pred_dict = {}
		with tf.Session() as sess:
			with tf.device("/" + str(self.args.dev)):
				# read model parameters from file
				saver.restore(sess, "save_val/model.ckpt")
				if self.mode=='u_p':
					# u_p: users to items
					count = 0
					pred_users = list(test_X['uid'].unique())
					print(len(pred_users))
					for user in pred_users:
						preds = []
						users = []
						pros = []
						count += 1
						if count%1000==0:
							print(count)
						# Only banks that have interactions with the user are currently considered
						banks = user_bank_dict[user]
						for bank in banks:
							pro_list = bank_pro_dict[bank].copy()
							# remove the positive product that appears in the train
							pro_list = self.difference_list(pro_list, user_pro_dict[user])
							user_list = [user]*len(pro_list)
							
							user_item_df = self.get_feed_data(user_list, pro_list, user_df, item_df)
							
							feed = {self.model._users: user_list, self.model._items_pos: pro_list, \
									self.model._pos_fea: user_item_df.iloc[:, 2:]}
							
							pred = sess.run(self.model._pred, feed)
							pred_list = list(pred)
							preds += pred_list
							users += user_list
							pros += pro_list
						
						# sort by predict score
						preds_sort, users_sort, pros_sort = self.sort_predictions(preds,users,pros)
						if bank==1:
							out_list.append(str(user) + ": " + str(pros_sort[:self.args.uti_k]))
							#out_list.append(str(prod) + ": " + str(pros_sort[:self.args.uti_k]))
						pred_dict[str(user)] = pros_sort[:self.args.uti_k]
				elif self.mode=='p_u':
					# p_u: items to users
					count = 0
					pred_prods = list(test_X['pid'].unique())
					print(len(pred_prods))
					for prod in pred_prods:
						count += 1
						#if count%100==0:
						#	print(count)
						user_list = list(user_pro_dict.keys()).copy()
						#random.shuffle(user_list)
						# remove the positive user that appears in the train
						user_list = self.difference_list(user_list, pro_user_dict[prod])
						pro_list = [prod]*len(user_list)
						
						user_item_df = self.get_feed_data(user_list, pro_list, user_df, item_df)
						
						feed = {self.model._users_pos: user_list, self.model._items: pro_list, self.model._pos_fea: user_item_df.iloc[:,2:], self.model._keep_prob: 1}
						
						pred = sess.run(self.model._pred, feed)
						pred_list = list(pred)
						
						# sort by predict score
						preds_sort, users_sort, pros_sort = self.sort_predictions(pred_list, user_list, pro_list)
						
						out_list.append(str(prod) + ": " + str(users_sort[:self.args.itu_k]))
						pred_dict[str(prod)] = users_sort[:self.args.itu_k]
		with codecs.open('data/predict/predictions', 'w', encoding = 'utf-8') as f:
			for val in out_list:
				f.write(val+"\n")
		return pred_dict
	
	def cold_start_test(self, test_count1_X, user_df, item_df):
		'''test the cold start
		test_count1_X: test data
		user_df, item_df: user and item features
		'''
		saver = tf.train.Saver()
		pred_dict = {}
		users = list(test_count1_X['uid'].unique())
		with tf.Session() as sess:
			with tf.device("/" + str(self.args.dev)):
				# read model parameters from file
				saver.restore(sess, "save_val/model.ckpt")
				pred_prods = list(test_count1_X['pid'].unique())
				for prod in pred_prods:
					item = prod
					user_list = users.copy()
					item_list = [item]*len(user_list)
					
					user_item_df = self.get_feed_data(user_list, item_list, user_df, item_df)
					
					feed = {self.model._users_pos: user_item_df['uid'], self.model._items: user_item_df['pid'], \
							self.model._pos_fea: user_item_df.iloc[:, 2:], self.model._keep_prob: 1}
					
					pred = sess.run(self.model._pred, feed)
					pred_list = list(pred)
					
					# sort by predict score
					preds_sort, users_sort, pros_sort = self.sort_predictions(pred_list, user_list, item_list)
					pred_dict[str(item)] = users_sort[:self.args.itu_k]
		return pred_dict
	
	def predict(self, users, items, user_df, item_df, out_file):
		'''predict
		users: a list of users
		items: a list of items
		user_df, item_df: user and item features
		out_file: write file path
		'''
		saver = tf.train.Saver()
		result_list = []
		with tf.Session() as sess:
			with tf.device("/" + str(self.args.dev)):
				# read model parameters from file
				saver.restore(sess, "save/model.ckpt")
				if self.mode=='u_p':
					for user in users:
						user_list = [user]*len(items)
						user_item_df = self.get_feed_data(user_list, items, user_df, item_df)
						
						feed = {self.model._users: user_list, self.model._items_pos: items, \
							self.model._pos_fea: user_item_df.iloc[:, 2:]}
					
						pred = sess.run(self.model._pred, feed)
						pred_list = list(pred)
						
						# sort by predict score
						preds_sort, users_sort, pros_sort = self.sort_predictions(pred_list, user_list, items)
						#result_list.append(str(user) + ": " + str(pros_sort[:self.args.uti_k]))
						result_list.append({str(user),str(pros_sort[:self.args.uti_k])})
				elif self.mode=='p_u':
					for item in items:
						item_list = [item]*len(users)
						user_item_df = self.get_feed_data(users, item_list, user_df, item_df)
						print(user_item_df.iloc[:, 2:])
						feed = {self.model._users_pos: user_item_df['uid'], self.model._items: item_list, \
							self.model._pos_fea: user_item_df.iloc[:, 2:], self.model._keep_prob: 1}
					
						pred = sess.run(self.model._pred, feed)
						pred_list = list(pred)
						pred_list = list(map(str, pred_list))
						
						# sort by predict score
						#preds_sort, users_sort, pros_sort = self.sort_predictions(pred_list, users, item_list)
						#result_list.append(str(item) + ": " + str(users_sort[:self.args.itu_k]))
						#result_list.append([item,users_sort[:self.args.itu_k]])
						result_line = str(item) + ','
						result_line += (',').join(pred_list)
						print(result_line)
						
						result_list.append(result_line)
		return result_list
		
	def difference_list(self, list, sub_list):
		'''to find the difference set of two lists
		'''
		list.extend(sub_list)
		temp = pd.DataFrame({'dul' : list})
		temp = temp.drop_duplicates(['dul'], keep=False)
		return temp['dul'].values.tolist()
	
	def get_feed_data(self, user_list, item_list, user_df, item_df):
		'''get the user features
		'''
		user_item_df = pd.DataFrame({"uid": user_list, "pid" : item_list})
		user_item_df = utils.feature_map(user_item_df, user_df, item_df)
		
		user_item_df['uid'] = user_item_df['uid'].apply(lambda x: self.args.num_users if x>self.args.num_users else x)
		user_item_df['pid'] = user_item_df['pid'].apply(lambda x: self.args.num_items if x>self.args.num_items else x)
		
		return user_item_df
	
	def sort_predictions(self, preds, users, pros):
		'''sort the predict products by scores
		'''
		pred_pack = [(pred,user,pro) for pred,user,pro in zip(preds,users,pros)]
		pred_pack.sort()
		pred_pack.reverse()
		preds_sort = [pred for pred,user,pro in pred_pack]
		users_sort = [user for pred,user,pro in pred_pack]
		pros_sort = [pro for pred,user,pro in pred_pack]
		return preds_sort, users_sort, pros_sort
	
	def cal_MRR(self, pred_dict, target):
		''' calculating MRR for user to item
		pred_dict: a dict, key is the mobile，value is a list of product sequence predicted
		target: the target dataframe for predict
		'''
		# real interactive data
		ground_truth = []
		for index,row in target.iterrows():
			ground_truth.append(str(row['pid']) + "\t" + str(row['uid']))
		return evaluation.MRR(pred_dict, ground_truth)
	
	def cal_pre_k(self, pred_dict, target):
		''' calculating precision@k for item to user
		pred_dict: a dict, key is the product，value is a list of user sequence predicted
		target: the target dataframe for predict
		'''
		# real interactive data
		ground_truth = []
		for index,row in target.iterrows():
			ground_truth.append(str(row['pid']) + "\t" + str(row['uid']))
		return evaluation.precision_k(pred_dict, ground_truth)