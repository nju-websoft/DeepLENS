'''
@file: train_test.py
@author: qxLiu
@time: 2020/3/8 16:00
'''

from deeplens import *
from dataset import *
# seed = 7

# def do_run(ds_name, topk, num_epoch, valid_step, mode='test', model_save_dir=OUT_MODEL_DIR, out_summ_dir, in_esbm_dir=None):
def do_run(ds_name, topk, num_epoch, valid_step, mode
		   ,in_embed_dir, in_esbm_dir
		   , model_save_dir, out_summ_dir):
	print('run on mode...',mode)
	dataset = DataSet(ds_name, topk, in_embed_dir=in_embed_dir, in_esbm_dir=in_esbm_dir)
	if os.path.isdir(model_save_dir):
		print('model samve dir exists!', model_save_dir)
	else:
		print('creating model save dir:', model_save_dir)
		os.mkdir(model_save_dir)
	test_favg_list = []
	valid_epoch_list = _read_epochs_from_log(ds_name, topk, model_save_dir) if mode=='test' else []
	for datafold in dataset.get_data_folds():
		if mode == 'train' or mode == 'all':
			model_name,valid_epoch, train_loss, valid_loss = _do_train_one_model(datafold, num_epoch, valid_step
								   ,model_save_dir=model_save_dir)
			valid_epoch_list.append(valid_epoch)
			# print('train fold %d:' % datafold.fold_id, valid_epoch, train_loss, valid_loss)
		if mode == 'test' or mode == 'all':
			if mode == 'test':
				valid_epoch = valid_epoch_list[datafold.fold_id]
			test_favg = _do_test(datafold, model_save_dir, valid_epoch, out_summ_dir, in_esbm_dir)
			test_favg_list.append(test_favg)
			# print('test fold %d:' % datafold.fold_id, test_favg)
	#==== out fold
	if mode == 'test' or mode == 'all':
		print(ds_name,topk.name, '5fold test avg:', np.mean(test_favg_list), test_favg_list)
	if mode == 'train' or mode == 'all': # log epochs
		log_file_path = os.path.join(model_save_dir, '{}_0log.txt'.format(model_name))
		with open(log_file_path,'a') as log_file:
			line = '{}-{} epoch:\t{}\n'.format(ds_name,topk.name, str(valid_epoch_list))
			log_file.write(line)

def _do_train_one_model(datafold, num_epoch, valid_step
						, model_save_dir=None, repeat_idx=0):
	stop_valid_epoch = None
	stop_valid_loss = None
	stop_train_loss = None
	tf.reset_default_graph()
	with tf.Session() as sess:
		# tf.set_random_seed(seed)
		model = DeepLENS(datafold.dim_triple, datafold.max_desc_size, fold_id=datafold.fold_id)
		model.set_sess(sess)
		sess.run(model.init)

		for epoch in range(num_epoch):
			_, train_loss = model.update(*datafold.get_train_input())
			print(epoch,'train_loss:',train_loss)
			if epoch%valid_step == 0:
				_, valid_loss = model.apply(*datafold.get_valid_input())
				if stop_valid_loss == None or valid_loss<stop_valid_loss:
					stop_valid_loss = valid_loss
					stop_valid_epoch = epoch
					stop_train_loss = train_loss
					if model_save_dir != None:
						model_save_path = os.path.join(
							model_save_dir, '{}_{}'.format(datafold.ds_name, datafold.topk.name),
							'{}_{}_{}_r{}_f{}_e{}'.format(model.name, datafold.ds_name, datafold.topk.name,
														  repeat_idx, datafold.fold_id, epoch))
						model.saver.save(sess, save_path=model_save_path)
						model.best_model_path = model_save_path

	print('train fold %d:' % datafold.fold_id, stop_valid_epoch, stop_train_loss, stop_valid_loss)
	return model.name, stop_valid_epoch, stop_train_loss, stop_valid_loss

def _do_test(datafold, model_save_dir, valid_epoch, out_summ_dir, in_esbm_dir, out_to_files=True):
	model = DeepLENS(datafold.dim_triple, datafold.max_desc_size
					 , fold_id=datafold.fold_id, do_build=False)
	tf.reset_default_graph()
	with tf.Session() as sess:
		# tf.set_random_seed(seed)
		model.init_for_test(sess, datafold, valid_epoch, model_save_dir)
		predict_tscores = model.apply(*datafold.get_test_input())
		# predict_tscores = np.squeeze(predict_tscores)
		tspan_list = datafold.test_eid_tspan_list
		tid_list = datafold.test_tid_list
		assert len(predict_tscores)==len(tid_list)

		eid_golds_dict = datafold.get_egolds_dict()
		favg_list = []
		# eid_tscorelist_dict = dict()
		for eidx, eid in enumerate(datafold.test_eid_list):
			tidx_start, tidx_end = tspan_list[eidx]
			tids = tid_list[tidx_start:tidx_end + 1]
			tscores = list(predict_tscores[tidx_start:tidx_end + 1])
			tid_tscore_list = list(zip(tids, tscores))
			tid_tscore_list.sort(key=lambda t: t[1], reverse=True)  # sort by tscore desc
			# print('eid:',eid, tid_tscore_list)
			summ_scored = tid_tscore_list[:datafold.topk.value]
			# print('summ_scored:',summ_scored)
			summ_tids, summ_scores = zip(*summ_scored)
			if out_to_files:
				gen_summ_file(datafold, eid, summ_tids, out_summ_dir=out_summ_dir, esbm_dir=in_esbm_dir)
			golds = eid_golds_dict.get(eid)
			favg = _eval_Fmeasure(summ_tids, golds)
			favg_list.append(favg)
	test_favg = np.mean(favg_list)
	print('test fold %d:' % datafold.fold_id, test_favg)
	return test_favg


def _eval_Fmeasure(summ_tids, gold_list):
	k = len(summ_tids)
	f_list = []
	for gold in gold_list:
		if len(gold) !=k:
			print('gold-k:',len(gold), k)
		assert len(gold)==k # for ESBM, not for dsFACES
		corr = len([t for t in summ_tids if t in gold])
		precision = corr/k
		recall = corr/len(gold)
		f1 = 2*precision*recall/(precision+recall) if corr!=0 else 0
		f_list.append(f1)
		# print('corr-prf:',corr,precision,recall,f1)
	favg = np.mean(f_list)
	# print('flist:',favg,f_list)
	return favg

def _read_epochs_from_log(ds_name, topk, model_save_dir, model_name=MODEL_NAME):
	log_file_path = os.path.join(model_save_dir, '{}_0log.txt'.format(model_name))
	key = '{}-{}'.format(ds_name, topk.name)
	epoch_list = None
	with open(log_file_path, 'r', encoding='utf-8') as f:
		for line in f:
			if line.startswith(key):
				epoch_list = list(eval(line.split('\t')[1]))
	return epoch_list

import argparse
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='DeepLENS: Deep Learning for Entity Summarization')
	# parser.add_argument("--ds_name", default='lmdb', type=str, help="string, 'dbpedia' or 'lmdb'")
	parser.add_argument("--mode", default='test', type=str, help="string, 'train', 'test' or 'all'")
	# parser.add_argument("--topk", default=TOPK.top10, type=lambda topK: TOPK[topK], help="string, 'top5' or 'top10'")
	parser.add_argument("--num_epoch", default=50, type=int, help="train model in total n epochs")
	parser.add_argument("--valid_step", default=1, type=int, help="num of epochs between two valid during training")
	parser.add_argument("--in_embed_dir", default=IN_EMBED_DIR, type=str,
						help="directory for input ids and embeddings")
	parser.add_argument("--in_esbm_dir", default=IN_ESBM_DIR, type=str,
						help="directory for input original data and golds")
	parser.add_argument("--model_save_dir", default=OUT_MODEL_DIR, type=str, help="directory for saving or restoring the trained models")
	parser.add_argument("--out_summ_dir", default=OUT_SUMM_DIR, type=str,
						help="directory for saving the generated summaries")

	args = parser.parse_args()
	# do_run(args.ds_name, args.topk, args.num_epoch, args.valid_step, mode=args.mode)
	for ds_name in ['dbpedia', 'lmdb']:
		for topk in [TOPK.top5, TOPK.top10]:
			do_run(ds_name, topk, args.num_epoch, args.valid_step
				   , mode=args.mode
				   , in_embed_dir=args.in_embed_dir, in_esbm_dir=args.in_esbm_dir
				   , model_save_dir=args.model_save_dir, out_summ_dir=args.out_summ_dir)