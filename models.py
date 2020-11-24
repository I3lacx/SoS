import config as cfg
import utils 

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np

import time
# Video and visualization stuff
import tqdm
# import PIL.Image, PIL.ImageDraw
# Might not need Ipython 
# from IPython.display import Image, HTML, clear_output
# import moviepy.editor as mvp
 #from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

# For exporting/importing? the model
import json
from google.protobuf.json_format import MessageToDict
from tensorflow.python.framework import convert_to_constants
# import matplotlib.pyplot as plt
# from visdom import Visdom


class CAModel(tf.keras.Model):
	""" Two modes, growing and classify, two completely seperate tasks """

	def __init__(self, channel_n=cfg.MODEL.CHANNEL_N, fire_rate=cfg.WORLD.CELL_FIRE_RATE,
							 add_noise=cfg.TRAIN.ADD_NOISE, env_size=cfg.WORLD.ENV_SIZE):
		# CHANNEL_N does *not* include the greyscale channel.
		# but it does include the 10 possible outputs.
		super().__init__()
		self.channel_n = channel_n
		self.fire_rate = fire_rate
		self.add_noise = add_noise
		self.env_size = env_size

		self.dmodel = self.get_dmodel()

		# Input Layer this works, zeroes would also work
		self(tf.keras.Input(shape=(cfg.WORLD.SIZE, cfg.WORLD.SIZE, cfg.MODEL.CHANNEL_N)))
		# self(tf.ones([1, 3, 3, channel_n + cfg.WORLD.ENV_SIZE]))

	def get_dmodel(self):
		""" returns dmodel based on cfg (number of layers and filter size)"""
		input_shape = (cfg.WORLD.SIZE, cfg.WORLD.SIZE, cfg.MODEL.CHANNEL_N)
		input_layer = tf.keras.Input(shape=input_shape)
		
		current_layer = Conv2D(cfg.MODEL.HIDDEN_FILTER_SIZE, 3, activation=tf.nn.relu,
								padding="SAME", name="perceive")(input_layer)


		previous_layer = current_layer
		for i in range(cfg.MODEL.HIDDEN_LAYERS):
			current_layer = Conv2D(cfg.MODEL.HIDDEN_FILTER_SIZE, 1, activation=None)(previous_layer)

			if cfg.MODEL.SKIP_CONNECTIONS:
				current_layer = tf.keras.layers.Add(name=f"skip_connection_{i}")([current_layer, previous_layer])

			if cfg.MODEL.BATCH_NORM:
				current_layer = tf.keras.layers.BatchNormalization()(current_layer)

			current_layer = tf.keras.layers.ReLU()(current_layer)
			previous_layer = current_layer


		if cfg.MODEL.LAST_LAYER_INIT == "ZEROS":
			final_layer = Conv2D(self.channel_n, 1, activation=None, kernel_initializer=tf.zeros_initializer)(previous_layer)
		else:
			final_layer = Conv2D(self.channel_n, 1, activation=None)(previous_layer)

		if cfg.MODEL.UPDATE_GATE:
			# Sigmoid because it has to be between 0 and 1
			w = Conv2D(1, 1, activation="sigmoid", name="update_gate")(input_layer)
			w = tf.keras.layers.Lambda(self.print_layer, arguments=dict(text="w"))(w)
			# TODO find more elegant solution:
			# self.update_gate = w

			# tf.print(w.numpy())
			one = tf.ones_like(w, dtype=tf.float32)
			l1 = tf.keras.layers.multiply([w, input_layer])
			negated_w = tf.keras.layers.Add()([one, tf.math.negative(w)])
			negated_w = tf.keras.layers.Lambda(self.print_layer, arguments=dict(text="neg_w"))(negated_w)
			# print(negated_w)
			
			# TODO not prefered option, this just changes the update, but otherwise val gets too high too quick!
			final_layer = tf.keras.layers.multiply([negated_w, final_layer])

			if cfg.MODEL.SET_WORLD_STATE:
				# This will add l1 to final layer and thus is setting the world state
				final_layer = tf.keras.layers.Add(name="Super_skip_layer")([l1, final_layer])
			# final_layer = tf.keras.layers.Lambda(self.print_layer, arguments=dict(text="final_layer"))(final_layer)

		model = tf.keras.models.Model(name="dmodel", inputs=input_layer, outputs=final_layer)

		return model

	# TODO this does't even make a difference, so just remove it?
	# @tf.function
	def call(self, x, fire_rate=None, manual_noise=None):
		env, state = tf.split(x, [self.env_size, self.channel_n], -1)

		ds = self.dmodel(x)

		# if cfg.MODEL.KEEP_OLD_STATE:
			# dmodel is its first and only layer...
			#tf.print(self.layers[0].get_layer("update_gate").output)



		#if cfg.MODEL_TASK == "growing":
		#	return x + self.test(x)

		if self.add_noise:
			if manual_noise is None:
				residual_noise = tf.random.normal(tf.shape(ds), 0., 0.02)
			else:
				residual_noise = manual_noise
			ds += residual_noise

		if fire_rate is None:
			fire_rate = self.fire_rate
		update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate

		# tf.print("state type", type(state))
		# TODO tested for env of size 1, this is hotfix to make it work for RGB
		# For now I don't need it. In case of Emoji growing, its 0
		if cfg.WORLD.TASK["TASK"] == "classify":
			living_mask = tf.math.reduce_sum(env, -1, keepdims=True) > 0.1
		elif cfg.WORLD.TASK["TASK"] == "growing":
			# TODO how does env play into the living mask?
			if cfg.WORLD.LIVING_MASK:
				living_mask = self.get_living_mask(state)
			else:
				living_mask = tf.constant(1)
			# living_mask = tf.math.reduce_sum(x[:,:,:,:3], -1, keepdims=True) >= 0
			# living_mask = tf.math.reduce_sum(x[:,:,:,:3], -1, keepdims=True) > 0.1
		else:
			raise ValueError()

		residual_mask = update_mask & living_mask

		# Directly assign the world state instead of updating it
		if cfg.MODEL.SET_WORLD_STATE:
			# In order to apply the mask, only the active channels will be updated
			# TODO this is a np operation, which should be converted to a tf operation, but not tested!
			# This will update all values in state where mask is True and set values to ds
			state = tf.where(residual_mask, ds, state)
			# np.putmask(state, residual_mask, ds)
			# state = ds
		else:
			ds *= tf.cast(residual_mask, tf.float32)
			state += ds

		return tf.concat([env, state], -1)

	# @tf.function
	def initialize(self, images):
		""" Expands Image with channel_n dim, filled with 0s """
		state = tf.zeros([tf.shape(images)[0], 28, 28, self.channel_n])
		images = tf.reshape(images, [-1, 28, 28, 1])
		return tf.concat([images, state], -1)


	def print_layer(self, x, text):
		# TODO config to enable/disable print
		if cfg.EXTRA.PRINT_LAYER:
			tf.print(text, x.shape, tf.math.reduce_min(x), tf.math.reduce_mean(x), tf.math.reduce_max(x))
		# tf.print(tf.math.reduce_max(x))
		return x

	# @tf.function
	def classify(self, x):
		""" Not perse classification, depends on the task. Returns lossable channels """
		if cfg.WORLD.TASK["TASK"] == "classify":
			# The last 10 layers are the classification predictions, one channel
			# per class. Keep in mind there is no "background" class,
			# and that any loss doesn't propagate to "dead" pixels.
			return x[:,:,:,-10:]
		elif cfg.WORLD.TASK["TASK"] == "growing":
			# First 4 Channels are RGBA
			return x[:,:,:,:4]
		else:
			raise ValueError(f"Task: {cfg.WORLD.TASK['TASK']} not implemented")

	@tf.function
	def get_living_mask(self, x):
		alpha = x[:, :, :, 3:4]
		return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], 'SAME') > 0.1


	def summary(self):
		# print("perceive: ", self.perceive.summary())
		print("------------")
		print("dmodel: ", self.dmodel.summary())

	# TODO add more model information in name
	def export_model(self, path):
		""" Export model as a json, for whatever reason """
		self.save_weights(path)

		cf = self.call.get_concrete_function(
				x=tf.TensorSpec([None, None, None, cfg.MODEL.CHANNEL_N+1]),
				fire_rate=tf.constant(0.5),
				manual_noise=tf.TensorSpec([None, None, None, cfg.MODEL.CHANNEL_N]))
		cf = convert_to_constants.convert_variables_to_constants_v2(cf)
		graph_def = cf.graph.as_graph_def()
		graph_json = MessageToDict(graph_def)
		graph_json['versions'] = dict(producer='1.14', minConsumer='1.14')
		model_json = {
				'format': 'graph-model',
				'modelTopology': graph_json,
				'weightsManifest': [],
		}
		with open(path+'.json', 'w') as f:
			json.dump(model_json, f)

# CAModel().perceive.summary()
# CAModel().dmodel.summary()

# TODO should this be here or in an evaluation file? Like a trainer?

# @title Evaluation functions
def eval_perform_steps(ca, x, yt, num_steps):
	yt_label = tf.argmax(yt, axis=-1)

	live_mask = x[..., 0] > 0.1
	live_mask_fl = tf.expand_dims(tf.cast(live_mask, tf.float32), -1)
	dead_channel = tf.cast(x[..., :1] <= 0.1, tf.float32)

	# for now the metric is aggregating everything.
	total_count = tf.reduce_sum(tf.cast(live_mask, tf.float32))

	avg_accuracy_list = []
	avg_total_agreement_list = []
	for _ in range(1, num_steps + 1):
		x = ca(x)

		y = ca.classify(x)
		y_label = tf.argmax(y, axis=-1)

		correct = tf.equal(y_label,  yt_label) & live_mask
		total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
		avg_accuracy_list.append((total_correct/total_count * 100).numpy().item())

		# agreement metrics
		# Important to exclude dead cells:
		y = y * live_mask_fl
		y_label_plus_mask = tf.argmax(tf.concat([y, dead_channel], -1), axis=-1)
		all_counts = []
		for idx in range(10):
			count_i = tf.reduce_sum(
					tf.cast(tf.equal(y_label_plus_mask, idx), tf.int32), axis=[1,2])
			all_counts.append(count_i)
		all_counts_t = tf.stack(all_counts, 1)
		# Now the trick is that if there is a total agreement, their sum is the same
		# as their max.
		equality = tf.equal(tf.reduce_max(all_counts_t, axis=1),
												tf.reduce_sum(all_counts_t, axis=1))
		sum_agreement = tf.reduce_sum(tf.cast(equality, tf.float32))
		avg_total_agreement_list.append(sum_agreement.numpy().item() / y.shape[0] * 100)

	return avg_accuracy_list, avg_total_agreement_list

def eval_batch_fn(ca, x_test, y_test, num_steps, mutate):
	x = ca.initialize(x_test)
	yt = y_test

	avg_acc_l_1, avg_tot_agr_l_1 = eval_perform_steps(ca, x, yt, num_steps)
	if not mutate:
		return avg_acc_l_1, avg_tot_agr_l_1
	# Accuracy after mutation!
	new_idx = np.random.randint(0, x_test.shape[0]-1, size=x_test.shape[0])
	new_x, yt = x_test[new_idx], y_test[new_idx]
	new_x = tf.reshape(new_x, [-1, 28, 28, 1])
	mutate_mask = tf.cast(new_x > 0.1, tf.float32)

	x = tf.concat([new_x, x[:,:,:,1:] * mutate_mask], -1)

	avg_acc_l_2, avg_tot_agr_l_2 = eval_perform_steps(ca, x, yt, num_steps)

	return avg_acc_l_1 + avg_acc_l_2, avg_tot_agr_l_1 + avg_tot_agr_l_2

