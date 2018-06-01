import numpy as np

def precision_k(pred_dict, ground_truth):
	''' precision @ k
	'''
	ground_truth_dict = {}
	for key_val in ground_truth:
		key = key_val.split('\t')[0]
		val = int(key_val.split('\t')[1])
		if key in ground_truth_dict:
			list = ground_truth_dict[key]
			list.append(val)
		else:
			list = []
			list.append(val)
			ground_truth_dict[key] = list
	
	result = []
	for key,val_list in pred_dict.items():
		if key in ground_truth_dict.keys():
			correct = 0
			for val in val_list:
				if val in ground_truth_dict[key]:
					correct += 1
			correct = correct/len(val_list)
			result.append(correct)
	return np.mean(result)


def MRR(pred_dict, ground_truth):
	''' Mean Reciprocal Rank
	pred_dict: a dict, key is the mobile，value is a list of product sequence predicted
	ground_truth: a list of target，format: "mobile + \t + product index"
	'''
	result = 0
	for key_val in ground_truth:
		key = key_val.split('\t')[0]
		val = int(key_val.split('\t')[1])
		if val in pred_dict[key]:
			pred_index = pred_dict[key].index(val)
			result += 1/(pred_index+1)
		else:
			result += 0
	return result/len(ground_truth)
	
if __name__ == '__main__':
	pred_dict = {'a':[1,3,5,7], 'b':[2,4,6,8]}
	ground_truth = []
	ground_truth.append("a\t3")
	ground_truth.append("a\t4")
	ground_truth.append("a\t5")
	ground_truth.append("b\t6")
	ground_truth.append("b\t7")
	ground_truth.append("b\t9")
	print(precision_k(pred_dict, ground_truth))