'''
@file: dataset.py
@author: qxLiu
@time: 2020/2/11 10:59
'''
import time
from enum import Enum
from dataprepare import *

CACHE_FOLD_DICT = dict() # key=ds_name+topk.name+str(fold_id)

class TOPK(Enum):
	top5 = 5;  # lef-side is name, right-side is value
	top10 = 10;

class DataSet(object):
	def __init__(self, ds_name, topk, num_folds=5):
		self.ds_name = ds_name
		self.topk = topk
		self.num_folds = num_folds

	def get_data_folds(self, reverse=False):
		trainList, validList, testList = get_5fold_train_valid_test_elist(self.ds_name) # eid_list, ordered asc
		key_prefix = ''.join([self.ds_name,self.topk.name])
		i_range = reversed(range(self.num_folds)) if reverse else range(self.num_folds)
		for i in i_range:
			key = ''.join([key_prefix, str(i)])
			datafold = CACHE_FOLD_DICT.get(key) # get from cache
			if datafold == None:
				datafold = DataFold(i, self.ds_name, self.topk, trainList[i], validList[i], testList[i])
				CACHE_FOLD_DICT[key] = datafold # cache
			yield datafold

	def get_data_fold(self, fold_id):
		key_prefix = ''.join([self.ds_name, self.topk.name])
		key = ''.join([key_prefix, str(fold_id)])
		datafold = CACHE_FOLD_DICT[key]
		return datafold

class DataFold(object):
	def __init__(self, fold_id, ds_name, topk, train_eid_list, valid_eid_list, test_eid_list):
		self.fold_id = fold_id
		self.ds_name = ds_name
		self.topk = topk
		self.train_eid_list = train_eid_list
		self.valid_eid_list = valid_eid_list
		self.test_eid_list = test_eid_list
		# for mmr
		self.tid_tembed_dict = dict()

		# loda data
		time_start = time.time()
		self._load_data_from_files()
		# print('finished DataFold._load_data_from_files(), time:', time.time()-time_start)
		# print('max_desc_size:',self.max_desc_size, ', dim_triple:', self.dim_triple)

		# for evaluation
		self.test_egolds_dict = None

	def _load_data_from_files(self):
		print('loading DataFold {}...'.format(self.fold_id)) #, ', DataFold._load_data_from_files()....')
		self.eid_tids_dict, self.max_desc_size = load_desc_tids(self.ds_name)
		tid_label_dict = load_tlabel(self.ds_name, self.topk.name)
		self.tembed_list, tid_idx_dict, self.dim_triple = load_embed(self.ds_name)
		assert self.dim_triple>0
		#===== for train
		self.train_eid_list.sort()
		self.train_desc_num = len(self.train_eid_list)
		self.train_desc_tnum_list = [len(self.eid_tids_dict.get(eid)) for eid in self.train_eid_list]
		self.train_tid_list = []
		self.train_eid_tspan_list = []
		idx_start = 0
		for i,eid in enumerate(self.train_eid_list):
			self.train_tid_list.extend(self.eid_tids_dict.get(eid))
			self.train_eid_tspan_list.append((idx_start, idx_start+self.train_desc_tnum_list[i]-1))
			idx_start += self.train_desc_tnum_list[i]
		self.train_tlabel_list = [tid_label_dict.get(tid) for tid in self.train_tid_list]
		self.train_t_embedidx_list = [tid_idx_dict.get(tid) for tid in self.train_tid_list]
		self.train_desc_t_embedidx_matrix = self._get_desc_t_embedidx_matrix(self.train_eid_tspan_list, self.train_t_embedidx_list)
		#===== for valid
		self.valid_eid_list.sort()
		self.valid_desc_num = len(self.valid_eid_list)
		self.valid_desc_tnum_list = [len(self.eid_tids_dict.get(eid)) for eid in self.valid_eid_list]
		self.valid_tid_list = []
		self.valid_eid_tspan_list = []
		idx_start = 0
		for i,eid in enumerate(self.valid_eid_list):
			self.valid_tid_list.extend(self.eid_tids_dict.get(eid))
			self.valid_eid_tspan_list.append((idx_start, idx_start+self.valid_desc_tnum_list[i]-1))
			idx_start += self.valid_desc_tnum_list[i]
		self.valid_tlabel_list = [tid_label_dict.get(tid) for tid in self.valid_tid_list]
		self.valid_t_embedidx_list = [tid_idx_dict.get(tid) for tid in self.valid_tid_list]
		self.valid_desc_t_embedidx_matrix = self._get_desc_t_embedidx_matrix(self.valid_eid_tspan_list, self.valid_t_embedidx_list)
		#===== for test
		self.test_eid_list.sort()
		self.test_desc_num = len(self.test_eid_list)
		self.test_desc_tnum_list = [len(self.eid_tids_dict.get(eid)) for eid in self.test_eid_list]
		self.test_tid_list = []
		self.test_eid_tspan_list = []
		idx_start = 0
		for i,eid in enumerate(self.test_eid_list):
			self.test_tid_list.extend(self.eid_tids_dict.get(eid))
			self.test_eid_tspan_list.append((idx_start, idx_start+self.test_desc_tnum_list[i]-1))
			idx_start += self.test_desc_tnum_list[i]
		# self.test_tlabel_list = [tid_label_dict.get(tid) for tid in self.test_tid_list]
		self.test_t_embedidx_list = [tid_idx_dict.get(tid) for tid in self.test_tid_list]
		self.test_desc_t_embedidx_matrix = self._get_desc_t_embedidx_matrix(self.test_eid_tspan_list, self.test_t_embedidx_list)

	def _get_desc_t_embedidx_matrix(self, eid_tspan_list, t_embedidx_list):
		desc_t_embedidx_matrix=[]

		for tid_start, tid_end in eid_tspan_list:
			desc_tnum = tid_end-tid_start+1
			pad_size = self.max_desc_size-desc_tnum
			desc_t_embedidx_list = t_embedidx_list[tid_start:tid_end+1] # item values correspond to idx of tembed_list
			desc_t_embedidx_list = np.pad(desc_t_embedidx_list, (0, pad_size), 'constant', constant_values=0)
			desc_t_embedidx_matrix.append(desc_t_embedidx_list)
			assert len(desc_t_embedidx_list)==self.max_desc_size
		desc_num = len(eid_tspan_list)
		assert len(desc_t_embedidx_matrix)==desc_num
		return desc_t_embedidx_matrix

	def get_train_input(self):
		return self.train_t_embedidx_list, self.tembed_list \
			, self.train_desc_t_embedidx_matrix, self.train_desc_tnum_list, self.train_desc_num \
			, self.train_tlabel_list
	def get_valid_input(self):
		return self.valid_t_embedidx_list, self.tembed_list \
			, self.valid_desc_t_embedidx_matrix, self.valid_desc_tnum_list, self.valid_desc_num \
			, self.valid_tlabel_list
	def get_test_input(self):
		return self.test_t_embedidx_list, self.tembed_list \
			, self.test_desc_t_embedidx_matrix, self.test_desc_tnum_list, self.test_desc_num


	def get_egolds_dict(self):
		'''
		for evaluate f-measure
		:return:
		'''
		if hasattr(self,'egolds_dict') and self.egolds_dict != None:
			return self.egolds_dict
		# print('loading golds...')
		self.egolds_dict = load_egolds_dict(self.ds_name, self.topk.name)
		return self.egolds_dict



