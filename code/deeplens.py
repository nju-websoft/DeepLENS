'''
@file: deeplens.py
@author: qxLiu
@time: 2020/2/8 17:44
'''
import os
import numpy as np
import tensorflow as tf

TF_FLOAT = tf.float32
TF_INT = tf.int32
MODEL_NAME = 'DeepLENS'

class DeepLENS():
	def __init__(self, dim_triple, size_desc
				 , mlpc_hidden_nodes = [64,64] # mlp for candidate triple
				 , mlpd_hidden_nodes = [64,64] # mlp for desc
				 , mlps_hidden_nodes = [64,64,64] # mlp for score
				 , learning_rate = 0.01
				 , hidden_layer_active=tf.nn.relu
				 , w_ini = tf.contrib.layers.xavier_initializer()
				 , b_ini = tf.contrib.layers.xavier_initializer()
				 , sess = None
				 , name = MODEL_NAME
				 , fold_id = None
				 , num_to_save = 1, restore_path = None
				 , do_build = True):

		self.sess = sess
		self.name = name
		self.fold_id = fold_id
		self.restore_path = restore_path

		# necessary configs:
		self.dim_triple = dim_triple  # for input size, dim_triple
		self.mlps_hidden_nodes = mlps_hidden_nodes
		self.hidden_layer_active = hidden_layer_active
		self.learning_rate = learning_rate
		# === almost fixed
		self._w_ini = w_ini
		self._b_ini = b_ini
		self.num_to_save = num_to_save

		# new attributes for m1
		self.SIZE_DESC = size_desc  # from datafold.max_desc_size
		self.mlpc_hidden_nodes = mlpc_hidden_nodes
		self.mlpd_hidden_nodes = mlpd_hidden_nodes
		assert self.mlpc_hidden_nodes[-1] == self.mlpd_hidden_nodes[-1]  # should have same encode output dim for attention cosine-sim

		# do init
		if do_build:
			self._build_model()
			self.saver = tf.train.Saver(max_to_keep=self.num_to_save)
			self.init = tf.initialize_all_variables()

	def print_key_params(self):
		return ', '.join(['mlpc_hidden_nodes: '+str(self.mlpc_hidden_nodes)
						,'mlpd_hidden_nodes: '+str(self.mlpd_hidden_nodes)
						,'mlps_hidden_nodes: '+str(self.mlps_hidden_nodes)
						  ])

	def set_sess(self, sess):
		self.sess = sess

	def _restore(self, sess):
		self._build_model()
		self.saver = tf.train.Saver(max_to_keep=self.num_to_save)
		self.init = tf.initialize_all_variables()
		self.sess = sess
		self.sess.run(self.init)
		print('restore from path:', self.best_model_path)
		self.saver.restore(self.sess, self.best_model_path)


	def _build_model(self):
		print('building model...', self.name)
		with tf.variable_scope(self.name+'_input-layer'):
			# for candidate triples
			self.t_embedidx_var = tf.placeholder(name='t_embedidx_list', shape=[None], dtype=TF_INT)  # feature vectors of triples, shape=[tripleNum]
			self.tembed_tensor = tf.placeholder(name='tembed_table', shape=[None, self.dim_triple], dtype=TF_FLOAT)  # feature vectors of triples, shape=[embedNum, featureSize]
			# for desc encode
			self.desc_t_embedidx_tensor = tf.placeholder(name='desc_t_embedidx', shape=[None, self.SIZE_DESC], dtype=TF_INT) # shape=(descnum, padding_size), prepend one special desc with tid=0
			self.desc_tnum_vector = tf.placeholder(name='desc_tnum', shape=[None], dtype=TF_FLOAT) # shape=(descnum)
			self.desc_num = tf.placeholder(name='desc_num', shape=(), dtype=tf.int32)  # scalar, desc+1
			# only for train
			self.label_list = tf.placeholder(name='grades',shape=[None], dtype=TF_FLOAT) # grade of each triple, shape=[tripleNum]

		with tf.variable_scope(self.name+'_t-encode-layer'):
			self.triple_tensor = tf.nn.embedding_lookup(self.tembed_tensor, self.t_embedidx_var)  # shape=(tnum, featureSize)
			ht_encode_out = self.triple_tensor
			for hid, num_unit in enumerate(self.mlpc_hidden_nodes):
				ht_name = ''.join(['ht_', str(hid)])
				hi = tf.layers.dense(ht_encode_out, num_unit
									 , activation=self.hidden_layer_active
									 , kernel_initializer=self._w_ini
									 , bias_initializer=self._b_ini
									 , name=ht_name)
				ht_encode_out=hi
		#==== m1: desc encode by mlp-attention
		hd_encode_extend_out = self._encode_desc(ht_encode_out) ## m11: desc encode by attention
		mlp_input = tf.concat([ht_encode_out  # self.triple_vec_list
								, hd_encode_extend_out], 1 )  # for each row, new_vec_dim = concat(act_vec, state_vec)

		with tf.variable_scope(self.name+'_mlp-layer'):
			mlp_encode_out = mlp_input
			# do concat : [hd, trans_expand_array]
			for hid, num_unit in enumerate(self.mlps_hidden_nodes):
				hmlp_name = ''.join(['hmlp_', str(hid)])
				hi = tf.layers.dense(mlp_encode_out, num_unit
									 , activation=self.hidden_layer_active
									 , kernel_initializer=self._w_ini
									 , bias_initializer=self._b_ini
									 , name=hmlp_name)
				mlp_encode_out=hi
			self.tembed_out = mlp_encode_out # for output embed
		with tf.variable_scope(self.name+'_output-layer'):
			h_out = tf.layers.dense(mlp_encode_out, 1) # linear unit, to predict grade
			self.h_out = tf.squeeze(h_out)  # from shape=(?,1) to shape=(?,), to have same shape as grades to compute loss
			self.loss = tf.losses.mean_squared_error(self.h_out, self.label_list)
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
			self.train_op = self.optimizer.minimize(self.loss)


	def _encode_desc(self, triple_encode_outs): # encode desc by attention
		with tf.variable_scope(self.name+'_desc_encode_layer'):
			hd_input = tf.nn.embedding_lookup(self.tembed_tensor,
											  self.desc_t_embedidx_tensor)  # shape: (e_num, desc_size, feature_size)
			# 1. mlp for each triple in desc
			for hid, num_unit in enumerate(self.mlpd_hidden_nodes):
				hd_name = ''.join(['hd_', str(hid)])
				hdi = tf.layers.dense(hd_input, num_unit
									  , activation=self.hidden_layer_active
									  , kernel_initializer=self._w_ini
									  , bias_initializer=self._b_ini
									  , name=hd_name)
				hd_input = hdi  # shape: (e_num, desc_size, num_unit)
			# print('hd_shape:', np.shape(hd_input))  # (?, 103, 64)
			# 2. do attention
			self.desc_vec_dim = self.mlpd_hidden_nodes[-1] # fnum_encoded
			# (1) expand triples: from shape(T_num, fnum_encoded) to shape(T_num, 1, fnum_encoded)
			triple_vec_list = tf.reshape(triple_encode_outs, [-1, 1, self.desc_vec_dim]) #
			# print('expand triples:',np.shape(triple_vec_list))
			# (2) expand descs by copy: from shape(e_num, desc_size, fnum_encoded) to shape(T_num, desc_size, fnum_encoded)
			desc_matrix_list = self._expand_encode(hd_input, self.desc_vec_dim, state_is_matrix=True, call_idx=0)
			# print('desc_matrix_list:',np.shape(desc_matrix_list))
			# (3) attention by cosine [Graves2014]: shape(T_num, desc_size)
			cosine_similarity_weights = tf.reduce_sum(triple_vec_list * desc_matrix_list, axis=-1) / (
				tf.norm(triple_vec_list, axis=-1) * tf.norm(desc_matrix_list, axis=-1)
				) # shape=(e_num, desc_size), (padding triples may cause nan)
			cosine_similarity_weights = tf.where(tf.is_nan(cosine_similarity_weights), tf.zeros_like(cosine_similarity_weights),
									  cosine_similarity_weights)  # replace nan to zeros
			cosine_similarity_weights = tf.nn.softmax(cosine_similarity_weights, axis=1) # softmax on attention
			# print('cosine_similarity_weights1:',np.shape(cosine_similarity_weights))
			# (4) use weight to desc: shape(T_num, desc_size, fnum_encoded)
			cosine_similarity_weights = tf.expand_dims(cosine_similarity_weights, axis=-1) # shape=(T_num, desc_size, 1)
			# print('cosine_similarity_weights2:',np.shape(cosine_similarity_weights))
			weighted_desc = tf.multiply(cosine_similarity_weights, desc_matrix_list) # shape=(T_num, desc_size, fnum_encoded)
			# print('weighted_desc:',np.shape(weighted_desc))
			# (5) do sum
			desc_vec_list = tf.reduce_sum(weighted_desc, axis=1, keepdims=True)  # for M11-2
			# print('desc_vec_list:',np.shape(desc_vec_list))
		# reshape: to shape(T_numm, fum_encoded)
		desc_vec_list = tf.reshape(desc_vec_list,[-1, self.desc_vec_dim])
		# print('desc_vec_list shape:',tf.shape(desc_vec_list))
		return desc_vec_list


	def _expand_encode(self, desc_vec_list, f_num, state_is_matrix=False, call_idx=0):# set call_idx for define different variables for multiple calls
		with tf.variable_scope(self.name + '_td_extend_layer_'+str(call_idx)):
			def cond(idx, trans_num, state_vec_list, trans_triple_num_list, trans_expand_array):
				return idx < trans_num

			def body(idx, trans_num, state_vec_list, trans_triple_num_list, trans_expand_array):
				trans_triple_num = trans_triple_num_list[idx]  # triple num of trans_idx
				if state_is_matrix:
					state_vec = state_vec_list[idx, :, :] # [desc_size, vec_dim]
					state_vec = [state_vec] # [1, desc_size, vec_dim]
					expand_matrix = tf.tile(state_vec, [trans_triple_num,1,1]) # expand by duplicate trans_triple_num times
					# size=[cand_num, desc_size, state_vec_dim] , copy state_vec cand_num times
				else:
					state_vec = state_vec_list[idx, 0, :] if f_num>1 else state_vec_list[idx,:] # [1, vec_dim]
					expand_op = tf.ones([trans_triple_num, 1], tf.float32)  # [cand_num, 1]
					expand_matrix = tf.matmul(expand_op, [state_vec])  # size=[cand_num, state_vec_dim] , copy state_vec cand_num times
				trans_expand_array = tf.cond(idx > 0, lambda: tf.concat([trans_expand_array, expand_matrix], axis=0), lambda: expand_matrix)

				idx = idx + 1
				return idx, trans_num, state_vec_list, trans_triple_num_list, trans_expand_array

			trans_expand_array = tf.Variable(tf.zeros([1, 1, f_num]),
											 expected_shape=[None, None, f_num] ,
											 trainable=False)  if state_is_matrix else tf.Variable(tf.zeros([1, f_num]),
											 expected_shape=[None, f_num] ,
											 trainable=False)

			# print('sss_trans_expand_array', trans_expand_array.get_shape())
			idx = tf.get_variable('encoder_idx', dtype=tf.int32, shape=[], initializer=tf.zeros_initializer());
			tensor_shape = tf.TensorShape([None,None, f_num]) if state_is_matrix else tf.TensorShape([None, f_num])
			self.idx_out, _, _, _, trans_expand_array = tf.while_loop(cond, body,
							[idx, self.desc_num, desc_vec_list, self.desc_tnum_vector, trans_expand_array]
							, shape_invariants=[idx.get_shape(), self.desc_num.get_shape(), desc_vec_list.get_shape()
							, self.desc_tnum_vector.get_shape(), tensor_shape])  #
			# print('arr_shape', trans_expand_array.get_shape())
			return trans_expand_array  # [desc_num * desc_candidate_num, vec_dim], state for each triple


	def update(self, t_embedidx_list, tembed_list
			   , desc_t_embedidx_matrix, desc_tnum_list, desc_num
			   , grade_list
			   ):
		# print('updaes:',np.shape(t_embedidx_list),np.shape(tembed_list)
		# 	  , type(desc_t_embedidx_matrix), np.shape(desc_tnum_list)
		# 	  , desc_num, np.shape(grade_list))
		feed_dict={self.t_embedidx_var: t_embedidx_list
					,self.tembed_tensor: tembed_list
					, self.desc_t_embedidx_tensor: desc_t_embedidx_matrix
					, self.desc_tnum_vector: desc_tnum_list
					, self.desc_num: desc_num
					, self.label_list: grade_list
			}
		predict_grades, loss, _ = self.sess.run([self.h_out, self.loss, self.train_op],feed_dict=feed_dict)
		return predict_grades, loss

	def apply(self, t_embedidx_list, tembed_list
			   , desc_t_embedidx_matrix, desc_tnum_list, desc_num # only used when self.do_encode_desc = True
			  , grade_list=None):
		if grade_list==None: # test
			feed_dict={self.t_embedidx_var: t_embedidx_list
					,self.tembed_tensor: tembed_list
					, self.desc_t_embedidx_tensor: desc_t_embedidx_matrix
					, self.desc_tnum_vector: desc_tnum_list
					, self.desc_num: desc_num
			}
			predict_grades = self.sess.run([self.h_out],feed_dict=feed_dict)
			predict_grades = np.squeeze(predict_grades)
			return predict_grades
		else: # valid
			feed_dict = {self.t_embedidx_var: t_embedidx_list
				, self.tembed_tensor: tembed_list
				, self.label_list: grade_list
				, self.desc_t_embedidx_tensor: desc_t_embedidx_matrix
				, self.desc_tnum_vector: desc_tnum_list
				, self.desc_num: desc_num
						 }
			predict_grades, loss= self.sess.run([self.h_out, self.loss],feed_dict=feed_dict)
			predict_grades = np.squeeze(predict_grades)
			return predict_grades, loss

	def init_for_test(self, sess, datafold, epoch, model_save_dir, repeat_idx=0):
		assert datafold.fold_id==self.fold_id

		self.best_epoch = epoch
		model_save_path = os.path.join(
			model_save_dir, '{}_{}'.format(datafold.ds_name, datafold.topk.name), '{}_{}_{}_r{}_f{}_e{}'.format(self.name, datafold.ds_name, datafold.topk.name,
														  repeat_idx, datafold.fold_id, epoch))
		self.best_model_path = model_save_path

		self._restore(sess)










