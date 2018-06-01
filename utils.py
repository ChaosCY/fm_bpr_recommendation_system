import numpy as np
import pandas as pd
import codecs
import random
from math import sqrt

def write_file(data_df, output_file):
	data_df.to_csv(output_file, header=None, index=False)

def data_to_csv(data_file):
	'''
	不考虑用户与产品交互次数，删除重复数据
	'''
	header = ['user', 'pid']
	data_df = pd.read_csv(data_file, names=header)
	data_df = data_df.drop_duplicates(['user','pid'])[['user', 'pid']]
	
	#用户交互产品为1的数据
	a = (data_df['user'].value_counts()==1).index
	b = (data_df['user'].value_counts()==1).values
	data_df1 = data_df[data_df['user'].isin(a[b])]
	
	#用户交互产品不为1的数据
	a = (data_df['user'].value_counts()!=1).index
	b = (data_df['user'].value_counts()!=1).values
	data_df2 = data_df[data_df['user'].isin(a[b])]
	
	#产品交互用户为1的数据
	a = (data_df['pid'].value_counts()<=1000).index
	b = (data_df['pid'].value_counts()<=1000).values
	data_df3 = data_df[data_df['pid'].isin(a[b])]
	
	#产品交互用户不为1的数据
	a = (data_df['pid'].value_counts()>1000).index
	b = (data_df['pid'].value_counts()>1000).values
	data_df4 = data_df[data_df['pid'].isin(a[b])]
	
	write_file(data_df1, "data/mid/data_u_count_1.csv")
	write_file(data_df2, "data/mid/data_u_count_2.csv")
	write_file(data_df3, "data/mid/data_p_count_1.csv")
	write_file(data_df4, "data/mid/data_p_count_2.csv")

def to_maps(data_df):
	'''
	获取用户-银行、用户-产品、银行-产品映射
	获取用户列表
	'''
		
	user_pro_map = data_df.groupby('user')['pid'].unique()
	users = list(user_pro_map.index)
	user_pro_dict = {}
	for user in users:
		user_pro_dict[user] = list(user_pro_map[user])
	
	pro_user_map = data_df.groupby('pid')['user'].unique()
	prods = list(pro_user_map.index)
	pro_user_dict = {}
	for prod in prods:
		pro_user_dict[prod] = list(pro_user_map[prod])
	
	return user_pro_dict, pro_user_dict

def to_train_file(data_df, times_neg_samples=0):
	'''get train file for the whole data
	'''
	data_df['eval'] = 1
	write_file(data_df, "data/train/train_all.csv")

def to_train_test(data_df, data_count1_df, train_file, test_file, test_count1_file, times_neg_samples=0):
	'''
	得到train和test集
	'''
	
	# 这里是将一些购买量非常少的产品类放到训练集
	a = (data_df['pid'].value_counts()>1000).index
	b = (data_df['pid'].value_counts()>1000).values
	
	# split train and test
	mobile_set = set()
	product_set = set()
	train_list = []
	test_list = []
	for index,row in data_df.iterrows():
		mobile = row['user']
		product = row['pid']
		temp_list = []
		line = ""
		for col in data_df.columns:
			temp_list.append(row[col])
			line = line + str(row[col]) + ","
		if mobile not in mobile_set:
			mobile_set.add(mobile)
			train_list.append(temp_list)
			continue
		if (product not in product_set) or (product not in a[b]):
			product_set.add(product)
			train_list.append(temp_list)
			continue
		rand = random.randint(0,9)
		if rand<=1:
			test_list.append(temp_list)
		else:
			train_list.append(temp_list)
	
	train_df = pd.DataFrame(train_list, columns=['user', 'pid'])
	test_df = pd.DataFrame(test_list, columns=['user', 'pid'])
	
	user_pro_dict, pro_user_dict = to_maps(train_df)
	
	write_file(train_df, train_file)
	write_file(test_df, test_file)
	write_file(data_count1_df, test_count1_file)

def neg_sample_bank_u_p(row, user_pro_dict, prods, neg_list):
	'''
	获取负样本（产品）
	'''
	user = row['uid']
	rand = random.sample(prods, 1)
	while rand in user_pro_dict[user]:
		rand = random.sample(prods, 1)
	
	temp_list = []
	temp_list.append(str(user))
	temp_list.append(int(rand[0]))
	temp_list.append("0")
	neg_list.append(temp_list)
	return neg_list

def neg_sample_bank_p_u(row, user_pro_dict, users, neg_list):
	'''
	获取负样本（用户）
	'''
	prod = row['pid']
	rand = random.sample(users, 1)
	while prod in user_pro_dict[rand[0]]:
		rand = random.sample(users, 1)
	
	temp_list = []
	temp_list.append(int(rand[0]))
	temp_list.append(str(prod))
	temp_list.append("0")
	neg_list.append(temp_list)
	return neg_list

def negative_sampling(data_df, user_pro_dict, pro_user_dict, times_neg_samples, mode):
	'''
	负采样
	'''
	neg_list = []
	users = list(user_pro_dict.keys())
	prods = list(pro_user_dict.keys())
	if times_neg_samples==0:
		return
	if mode=='u_p':
		series = data_df.apply(lambda row : neg_sample_bank_u_p(row, user_pro_dict, prods, neg_list), axis=1).reset_index(drop=True)
	elif mode=='p_u':
		series = data_df.apply(lambda row : neg_sample_bank_p_u(row, user_pro_dict, users, neg_list), axis=1).reset_index(drop=True)
	neg_df = pd.DataFrame(series[len(series)-1], columns=['uid','pid','eval'])
	return neg_df

def read_data(train_file):
	'''
	读取数据
	'''
	header = ['user', 'pid']
	train_df = pd.read_csv(train_file, names=header)
	
	user_index_map = pd.DataFrame(train_df['user'].unique())
	user_index_map.columns = ['user']
	user_index_map['uid'] = user_index_map.index
	train_df = pd.merge(train_df, user_index_map, left_on='user', right_on='user', how='left')
	del train_df['user']
	train_df['pid'] = train_df['pid'].astype('int64')
	train_df['eval'] = 1
	
	train_df.rename(columns={'uid':'user'}, inplace = True)
	user_pro_dict, pro_user_dict = to_maps(train_df)
	train_df.rename(columns={'user':'uid'}, inplace = True)
	print(train_df.head())
	train_X = train_df[['uid','pid']]
	train_y = train_df['eval']
	
	return train_X, train_y, user_pro_dict, pro_user_dict, user_index_map

def read_data_val(train_file, test_file, test_count1_file):
	'''
	读取数据（测试）
	'''
	header = ['user', 'pid']
	train_df = pd.read_csv(train_file, names=header)
	test_df = pd.read_csv(test_file, names=header)
	test_count1_df = pd.read_csv(test_count1_file, names=header)
	train_df['eval'] = 1
	test_df['eval'] = 1
	test_count1_df['eval'] = 1
	
	df_all = train_df.append(test_df)
	df_all = df_all.append(test_count1_df)
	user_index_map = pd.DataFrame(df_all['user'].unique())
	user_index_map.columns = ['user']
	user_index_map['uid'] = user_index_map.index
	
	train_df = pd.merge(train_df, user_index_map, left_on='user', right_on='user', how='left')
	del train_df['user']
	test_df = pd.merge(test_df, user_index_map, left_on='user', right_on='user', how='left')
	del test_df['user']
	test_count1_df = pd.merge(test_count1_df, user_index_map, left_on='user', right_on='user', how='left')
	del test_count1_df['user']
	
	train_df.rename(columns={'uid':'user'}, inplace = True)
	user_pro_dict, pro_user_dict = to_maps(train_df)
	train_df.rename(columns={'user':'uid'}, inplace = True)
	
	train_X = train_df[['uid','pid']]
	train_y = train_df['eval']
	
	test_X = test_df[['uid','pid']]
	test_y = test_df['eval']
	
	test_count1_X = test_count1_df[['uid','pid']]
	test_count1_y = test_count1_df['eval']
	
	return train_X, train_y, test_X, test_y, test_count1_X, test_count1_y, user_pro_dict, pro_user_dict, user_index_map
	
def read_user_data(user_file, user_index_map):
	header = ['user','sex','location','channel','purchase_rank','property_rank','property_rank_province','economic_flow_stability','asset_stability','balance','invest_out','invest_in']
	user_df = pd.read_csv(user_file, names=header)
	user_df = pd.merge(user_df, user_index_map, left_on='user', right_on='user', how='left')
	del user_df['user']
	return user_df

def read_item_data(item_file):
	item_df = pd.read_csv(item_file)
	item_df = item_df[['pid','type1','type2','type3','risk','term','yield','amount','currency','fund_type0','fund_type1','fund_type2','fund_type3','fund_type4','fund_type5','fund_type6','fund_type7','fund_type8','fund_type9']]
	return item_df
	
def feature_map(ex_df, user_df, item_df):
	ex_df[["uid"]] = ex_df[["uid"]].astype(int)
	item_map_df = pd.merge(ex_df, item_df, left_on='pid', right_on='pid', how='left')
	item_map_df.fillna(0, inplace=True)
	item_user_map_df = pd.merge(item_map_df, user_df, left_on='uid', right_on='uid', how='left')
	item_user_map_df.fillna(0, inplace=True)
	return item_user_map_df

if __name__ == '__main__':
	train_file = 'data/train/train.csv';
	test_file = 'data/test/test.csv'
	test_count1_file = 'data/test/test_count1.csv'
	train_X, train_y, test_X, _, test_count1_X, _, user_bank_dict, user_pro_dict, bank_pro_dict, bank_user_dict, pro_user_dict, pro_bank_dict = read_data(train_file, test_file, test_count1_file, False)
	pro_file = 'data/mid/licai_features.csv'
	user_file = 'data/mid/user_features.csv'
	prod_df = read_pro_data(pro_file)
	user_df = read_user_data(user_file)
	print(feature_map(test_X, prod_df, user_df).head())