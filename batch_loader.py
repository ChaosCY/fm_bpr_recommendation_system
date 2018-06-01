import numpy as np
import math

class BatchLoader(object):
	'''
	generate batches, return every batch
	'''
	def __init__(self, batch_size, data_X, data_y):
		self.batch_size = batch_size
		#data_X, data_y = self.copy_samples(data_X, data_y)
		self.create_batches(data_X, data_y)
		
	def create_batches(self, data_X, data_y):
		'''
		generate batches
		'''
		self.num_batches = math.ceil(len(data_X)/self.batch_size)
		
		if self.num_batches == 0:
			assert False, "Not enough data."
		
		num_samples = len(data_X)
		indexs = np.arange(num_samples)
		#np.random.shuffle(indexs)
		
		xdata = data_X.iloc[indexs]
		ydata = data_y.iloc[indexs]
		self.x_batches = np.array_split(xdata, self.num_batches)
		self.y_batches = np.array_split(ydata, self.num_batches)

	def next_batch(self):
		'''
		get the next batch
		'''
		x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
		self.pointer += 1
		return x, y
		
	def reset_batch_pointer(self):
		'''
		when an epoch is finished, reset the batch pointer to 0
		'''
		self.pointer = 0
	
	def copy_samples(self, data_X, data_y):
		data_X['eval'] = data_y
		
		user_appear_count = data_X['uid'].value_counts()
		user_less = list(user_appear_count[user_appear_count.values<=3].index)
		user_append = data_X[data_X['uid'].isin(user_less)]
		data_X = data_X.append([user_append])
		
		item_appear_count = data_X['pid'].value_counts()
		item_less = list(item_appear_count[item_appear_count.values<=3].index)
		item_append = data_X[data_X['pid'].isin(item_less)]
		data_X = data_X.append([item_append])
		
		data_y = data_X['eval']
		del data_X['eval']
		
		return data_X, data_y