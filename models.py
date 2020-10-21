import config as cfg

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np

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

	def __init__(self, channel_n=cfg.CHANNEL_N, fire_rate=cfg.CELL_FIRE_RATE,
							 add_noise=cfg.ADD_NOISE, env_size=cfg.ENV_SIZE):
		# CHANNEL_N does *not* include the greyscale channel.
		# but it does include the 10 possible outputs.
		super().__init__()
		self.channel_n = channel_n
		self.fire_rate = fire_rate
		self.add_noise = add_noise
		self.env_size = env_size

		self.perceive = tf.keras.Sequential([
					Conv2D(80, 3, activation=tf.nn.relu, padding="SAME"),
			])  # 80 filters, why 80?

		self.dmodel = tf.keras.Sequential([
					Conv2D(80, 1, activation=tf.nn.relu), # 1x1 conv 80 -> 80 is fully connected but shared parameters?
					Conv2D(self.channel_n, 1, activation=None,   # same here, but channel_n output and no activation
											 kernel_initializer=tf.zeros_initializer),  # TODO why zeros?
					# Init by zeros because it helps learning. As if its random, the optimizer has to fight
					# to remove badly learned patterns, but starting at 0 is easier to learn
		])

		# TODO this seems like not good, maybe tf1 usage of tf2?
		self(tf.zeros([1, 3, 3, channel_n + 1]))  # dummy calls to build the model

	# @call is like tf.function(call), but nicer
	# tf function: "compiles a function into a callable TF graph"
	# call is just one simulation step?
	@tf.function
	def call(self, x, fire_rate=None, manual_noise=None):
		env, state = tf.split(x, [self.env_size, self.channel_n], -1)
		ds = self.dmodel(self.perceive(x))
		if self.add_noise:
			if manual_noise is None:
				residual_noise = tf.random.normal(tf.shape(ds), 0., 0.02)
			else:
				residual_noise = manual_noise
			ds += residual_noise

		if fire_rate is None:
			fire_rate = self.fire_rate
		update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate

		# TODO tested for env of size 1, this is hotfix to make it work for RGB
		living_mask = tf.math.reduce_sum(env, -1, keepdims=True) > 0.1

		residual_mask = update_mask & living_mask
		ds *= tf.cast(residual_mask, tf.float32)
		state += ds
		
		return tf.concat([env, state], -1)

	@tf.function
	def initialize(self, images):
		""" Expands Image with channel_n dim, filled with 0s """
		state = tf.zeros([tf.shape(images)[0], 28, 28, self.channel_n])
		images = tf.reshape(images, [-1, 28, 28, 1])
		return tf.concat([images, state], -1)

	@tf.function
	def classify(self, x):
		# The last 10 layers are the classification predictions, one channel
		# per class. Keep in mind there is no "background" class,
		# and that any loss doesn't propagate to "dead" pixels.
		return x[:,:,:,-10:]

	def summary(self):
		print("perceive: ", self.perceive.summary())
		print("------------")
		print("dmodel: ", self.dmodel.summary())

	def export_model(self, path):
		""" Export model as a json, for whatever reason """
		self.save_weights(path)

		cf = self.call.get_concrete_function(
				x=tf.TensorSpec([None, None, None, cfg.CHANNEL_N+1]),
				fire_rate=tf.constant(0.5),
				manual_noise=tf.TensorSpec([None, None, None, cfg.CHANNEL_N]))
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

