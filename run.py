import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import codecs

from model import Model
from batch_loader import BatchLoader
import utils

tf.app.flags.DEFINE_string('mode', None, 'validaion or train or predict')
FLAGS = tf.app.flags.FLAGS

def main():
	# setting parameters
	parser = argparse.ArgumentParser(
						formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--num_factors', type=float, default=10,
						help='embedding size')
	parser.add_argument('--model', type=str, default='mlp_bpr',
						help='model')
	parser.add_argument('--epoches', type=str, default=1,
						help='epoches')
	parser.add_argument('--learning_rate', type=float, default=0.01,
						help='learning rate')
	parser.add_argument('--reg_lambda', type=float, default=1.0,
						help='l2_regularizer lambda')
	parser.add_argument('--layers', nargs='?', default='[10,1]',
						help="Size of each layer.")
	parser.add_argument('--batch_size', type=int, default=10000,
						help='minibatch size')
	parser.add_argument('--recom_mode', type=str, default='p_u',
						help='recommendation mode, u_p: users to items, p_u: items to users')
	parser.add_argument('--decay_rate', type=float, default=0.99,
						help='decay rate for Adam')
	parser.add_argument('--keep_prob', type=float, default=0.2,
						help='dropout probility')
	parser.add_argument('--uti_k', type=int, default=30,
						help='top-k recommendation for recommending items to user')
	parser.add_argument('--itu_k', type=int, default=100,
						help='top-k recommendation for recommending users to item')
	parser.add_argument('--log_dir', type=str, default='logs',
						help='directory to store tensorboard logs')
	parser.add_argument('--mode', type=str, default='validation',
						help='train: only train the model, validation: train the model and test it with test data, predict: predict new data')
	parser.add_argument('--dev', type=str, default='cpu',
						help='training by CPU or GPU, input cpu or gpu:0 or gpu:1 or gpu:2 or gpu:3')
	parser.add_argument('--pIter', type=int, default=2,
						help='how many rounds of iterations show the effect on the test set')
	args = parser.parse_args()
	
	if FLAGS.mode=='validation':
		# read data from file
		train_file = 'data/train/train_u.csv'
		test_file = 'data/test/test_u.csv'
		test_count1_file = 'data/test/test_u_count1.csv'
		
		train_X, train_y, test_X, test_y, test_count1_X, test_count1_y, user_pro_dict, pro_user_dict, user_index_map = utils.read_data_val(train_file, test_file, test_count1_file)
		print(test_count1_X)
		# read feature data from file
		user_file = 'data/mid/user_features.csv'
		item_file = 'data/mid/item_features.csv'
		user_feature_df = utils.read_user_data(user_file, user_index_map)
		item_feature_df = utils.read_item_data(item_file)
		
		# generate batches
		batch_loader = BatchLoader(args.batch_size, train_X, train_y)
		
		args.num_users = np.max([np.max(train_X['uid'])]) + 1
		args.num_items = np.max([np.max(train_X['pid'])]) + 1
		
		model = Model(args, args.recom_mode, batch_loader)
		
		model.train_val(test_X, user_pro_dict, pro_user_dict, user_feature_df, item_feature_df)
		#pred_dict = model.test(test_X, user_pro_dict, pro_user_dict, user_feature_df, item_feature_df)
		#if args.recom_mode=='u_p':
		#	print(model.cal_MRR(pred_dict, test_X))
		#elif args.recom_mode=='p_u':
		#	print(model.cal_pre_k(pred_dict, test_X))
		
		pred_dict = model.cold_start_test(test_count1_X, user_feature_df, item_feature_df)
		print(model.cal_pre_k(pred_dict, test_count1_X))
	elif FLAGS.mode=='train':
		# read data from file
		train_file = 'data/train/train_u.csv'
		train_X, train_y, user_pro_dict, pro_user_dict, user_index_map = utils.read_data(train_file)
		# read feature data from file
		user_file = 'data/mid/user_features.csv'
		item_file = 'data/mid/item_features.csv'
		user_feature_df = utils.read_user_data(user_file, user_index_map)
		item_feature_df = utils.read_item_data(item_file)
		
		# generate batches
		batch_loader = BatchLoader(args.batch_size, train_X, train_y)
		 
		args.num_users = np.max(train_X['uid']) + 1
		args.num_items = len(item_feature_df) + 1
		model = Model(args, args.recom_mode, batch_loader)
		
		# train and save models
		model.train(user_pro_dict, pro_user_dict, user_feature_df, item_feature_df)
		# save the user and index
		utils.write_file(user_index_map['user'], "save/user_index_map")
		
	elif FLAGS.mode=='predict':
		user_index_file = 'save/user_index_map'
		user_index_map = pd.read_csv(user_index_file, names=['user'])
		args.num_users = len(user_index_map)
		user_file = 'data/data_pred/user_file.txt'
		#user_file = 'data/mid/user_features.csv.liaoning'
		user_df = pd.read_csv(user_file, names=['user'])
		#user_df = pd.read_csv(user_file,header=None)
		#user_df.rename(columns={0: 'user'}, inplace=True)
		#user_df = user_df['user']
		
		
		item_file = 'data/data_pred/item_file.txt'
		item_df = pd.read_csv(item_file, names=['pid'])
		
		user_index_map = user_index_map.append(user_df[['user']]).drop_duplicates(['user']).reset_index(drop=True)
		user_index_map['uid'] = user_index_map.index
		
		user_df = pd.merge(user_df, user_index_map, left_on='user', right_on='user', how='left')
		#del user_df['user']
		
		# read feature data from file
		user_file = 'data/mid/user_features.csv'
		item_file = 'data/mid/item_features.csv'
		user_feature_df = utils.read_user_data(user_file, user_index_map)
		item_feature_df = utils.read_item_data(item_file)
		args.num_items = len(item_feature_df) + 1
		
		out_file = 'data/predict/predictions'
		model = Model(args, args.recom_mode)
		result_list = model.predict(user_df['uid'].values.tolist(), item_df['pid'].values.tolist(), user_feature_df, item_feature_df, out_file)
		user_list = user_df['user'].values.tolist()
		head_line = "cate_id,"
		head_line += (',').join(user_list)
		result_list.insert(0, head_line)
		with codecs.open(out_file, 'w', encoding = 'utf-8') as f:
			for val in result_list:
				f.write(val+"\n")
	else:
		print('incorrect mode input...')
	

if __name__ == '__main__':
	main()