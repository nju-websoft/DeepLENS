'''
@file: dataprepare.py
@author: qxLiu
@time: 2020/3/4 10:45
'''
import numpy as np
import os
import os.path as path
IN_ESBM_DIR = os.path.join(path.dirname(os.getcwd()), 'data', 'in_ESBM_benchmark_v1.2')
IN_EMBED_DIR = os.path.join(path.dirname(os.getcwd()),'data','in_embed')
OUT_MODEL_DIR = os.path.join(path.dirname(os.getcwd()),'data','out_model')
OUT_SUMM_DIR = os.path.join(path.dirname(os.getcwd()),'data','out_summ')

#============ in

def get_5fold_train_valid_test_elist(ds_name_str, esbm_dir=IN_ESBM_DIR):
	if ds_name_str == "dbpedia":
		split_path = path.join(esbm_dir, "dbpedia_split")
	elif ds_name_str == "lmdb":
		split_path = path.join(esbm_dir, "lmdb_split")
	elif ds_name_str == "dsfaces":
		split_path = path.join(esbm_dir, "dsfaces_split")
	else:
		raise ValueError("The database's name must be dbpedia or lmdb")

	trainList, validList, testList = [],[],[]
	for i in range(5): # 5-folds
		# read split eid files
		fold_path = path.join(split_path, 'Fold'+str(i))
		train_eids = _read_split(fold_path,'train')
		valid_eids = _read_split(fold_path,'valid')
		test_eids = _read_split(fold_path,'test')
		trainList.append(train_eids)
		validList.append(valid_eids)
		testList.append(test_eids)
	return trainList, validList, testList



def _read_split(fold_path, split_name):
	'''
	:param fold_path:
	:param split_name: 'train', 'valid', 'test'
	:param eid_idx_dict:
	:param data:
	:param label:
	:return:
	'''
	split_eids = []
	with open(path.join(fold_path, "{}.txt".format(split_name)),encoding='utf-8') as f:
		for line in f:
			if len(line.strip())==0:
				continue
			eid = int(line.split('\t')[0])
			split_eids.append(eid)
	return split_eids





def load_desc_tids(ds_name_str, embed_dir=IN_EMBED_DIR):
	in_file = os.path.join(embed_dir, '{}_tids.txt'.format(ds_name_str))
	max_desc_size = None
	eid_tids_dict = dict()
	with open(in_file, 'r', encoding='utf-8') as f:
		for line in f:
			items = line.strip().split('\t')
			eid = int(items[0])
			tids = eval(items[1])
			eid_tids_dict[eid]=tids
			desc_size = len(tids)
			if max_desc_size is None or desc_size>max_desc_size:
				max_desc_size = desc_size
	# # for observe
	# print('size:', max_desc_size)
	# for k,v in eid_tids_dict.items():
	# 	print('k:',k,type(k),'v:',type(v),len(v))
	return eid_tids_dict, max_desc_size



def load_embed(ds_name_str, in_embed_dir=IN_EMBED_DIR):
	embed_path = path.join(in_embed_dir, '{}_fastText_vec.npz'.format(ds_name_str))
	d = np.load(embed_path)
	tembed_dict = eval(str(d['pvembedding_ftaw']))
	#=== convert format
	tid_idx_dict = dict(zip(tembed_dict.keys(), range(len(tembed_dict)))) # dict<tid, t_embed_idx>
	tembed_list = [tembed_dict.get(tid) for tid in tembed_dict.keys()]
	dim_triple = len(tembed_dict.get(0)) # 600
	# #=== check usage
	# print(tembed_dict.get(0)) # zeros
	# for k,v in tembed_dict.items():
	# 	print('k:',k,type(k), 'v:',type(v),len(v))
	return tembed_list, tid_idx_dict, dim_triple


def load_tlabel(ds_name_str, topk_str, in_embed_dir=IN_EMBED_DIR):
	in_tlabel_file = os.path.join(in_embed_dir, '{}_tlabels_{}.txt'.format(ds_name_str, topk_str))
	tid_label_dict = dict()
	with open(in_tlabel_file, 'r',encoding='utf-8') as f:
		for line in f:
			items = line.strip().split('\t')
			tid = int(items[0])
			tlabel = int(items[1])
			tid_label_dict[tid] = tlabel
	# # # #=== check usage
	# print(tid_label_dict.get(0)) # None
	# for k,v in tid_label_dict.items():
	# 	print('k:',k,type(k), 'v:',type(v),v)
	return tid_label_dict

def load_egolds_dict(ds_name_str, topk_str, embed_dir=IN_EMBED_DIR):
	in_path = os.path.join(embed_dir, '{}_egolds_{}.txt'.format(ds_name_str, topk_str))
	eid_golds_dict = dict()
	with open(in_path, 'r', encoding='utf-8') as f:
		for line in f:
			items = line.strip().split('\t')
			tid = int(items[0])
			golds = eval(items[1])
			eid_golds_dict[tid] = golds
	return eid_golds_dict

#============= out
def gen_summ_file(datafold, eid, summ_tids, out_summ_dir=OUT_SUMM_DIR, esbm_dir=IN_ESBM_DIR):
	desc_tids = datafold.eid_tids_dict.get(eid)
	summ_tidxs = [desc_tids.index(tid) for tid in summ_tids]
	in_file = os.path.join(esbm_dir, '{}_data'.format(datafold.ds_name), str(eid), '{}_desc.nt'.format(eid))
	desc_lines = []
	with open(in_file, 'r', encoding='utf-8') as inf:
		for triple in inf:
			if len(triple.strip())>0:
				desc_lines.append(triple)
	summ_lines = [desc_lines[idx] for idx in summ_tidxs]
	out_e_dir = os.path.join(out_summ_dir, datafold.ds_name, str(eid))
	if not os.path.isdir(out_e_dir):
		os.makedirs(out_e_dir)
	out_file = os.path.join(out_e_dir, '{}_{}.nt'.format(eid,datafold.topk.name))
	print('output:',out_file)
	with open(out_file, 'w', encoding='utf-8') as outf:
		outf.writelines(summ_lines)





if __name__ == '__main__':
	results = get_5fold_train_valid_test_elist('dbpedia')
	print('train:', results[0])
	print('valid:', results[1])
	print('test:', results[2])