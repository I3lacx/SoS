"""
File to have all training functions and such
Also for testing?
"""
# Maybe also add the cfg dict as input to set configurations
import config as cfg
import utils
import datasets
import display as disp
import models

import tensorflow as tf
import numpy as np


class Trainer:
	""" 
	Trainer class to handle most of the training stuff, using the ca and the dataset
	cfg should configure loss function, batch size and model configs.
	Could be set in here, but so far in config file.
	"""

	def __init__(self, x_train, y_train, ca, vis):
		self.x_train = x_train
		self.y_train = y_train
		self.ca = ca

		# Visdom for visualization
		self.vis = vis

		# mostly to set the shape, values will be overwirtten anyways
		# TODO Variable does not really help, just makes it more complicated
		self.x = tf.Variable(self.x_train[0:cfg.TRAIN.BATCH_SIZE], trainable=False)
		self.y = tf.Variable(self.y_train[0:cfg.TRAIN.BATCH_SIZE], trainable=False)
		# How much initial seeds used inside the pool, decreases after steps

		# Set tf Variables
		self.loss_log = []
		# TODO only for debugging, maybe not the best spot here, not sure how though
		self.grad_log = []
		self.pool_log = []
		self.x_log = []

		self.seed_idx = tf.Variable(0, dtype=tf.int32, trainable=False)
		self.seed_idx.assign(self.get_seed_idx())

		# To set current number of ca steps
		self.num_ca_steps = tf.Variable(0, dtype=tf.int32, trainable=False)

		# Will be used if USE_PATTERN_POOL is true
		self.batch = None

		self.optimizer = self.get_tf_optimizer()
		self.pool = self.get_pool()

		# Get function to apply gradients
		self.apply_grads = self.get_apply_grad_fn()

		# TODO remove for testing only
		self.pool_t = np.repeat(self.x_train[0:1], cfg.TRAIN.POOL_SIZE, 0)
		# print("pool size ", self.pool_t.shape)


	def save_weights(self, path="", name="model"):
		if path == "":
			path = utils.get_full_log_path()

		utils.ensure_dir(path)

		self.ca.save_weights(path + name)

	def load_weights(self, path):
		self.ca.load_weights(path)

	def get_tf_optimizer(self):
		# TODO change lr/optimizer
		lr = cfg.TRAIN.LR
		warmup_steps = cfg.TRAIN.WARM_UP * cfg.TRAIN.NUM_TRAIN_STEPS
		# TODO dependent on warmup config
		lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
			[warmup_steps, 1000], [lr*0.1, lr, lr*0.1])
		trainer = tf.keras.optimizers.Adam(lr)
		return trainer

	def get_pool(self):
		if cfg.WORLD.TASK["TASK"] == "growing":
			# Use x,y directly as sample pool (always the same)
			# x and y train are created for pool size
			pool = datasets.SamplePool(x=self.x_train, y=self.y_train)
		elif cfg.WORLD.TASK["TASK"] == "classify":
			starting_indexes = np.random.randint(0, self.x_train.shape[0]-1, size=cfg.TRAIN.POOL_SIZE)
			# pool is a Class defined in datasets
			pool = datasets.SamplePool(x=self.ca.initialize(self.x_train[starting_indexes]).numpy(),
						 y=self.y_train[starting_indexes])
		else:
			raise ValueError()

		return pool

	def get_random_train_idx(self, size):
		return np.random.randint(0, self.x_train.shape[0]-1, size=size)

	@utils.timeit
	def get_new_x_y(self):
		if cfg.WORLD.TASK["TASK"] == "classify":
			return self.get_new_mnist_x_y()
		elif cfg.WORLD.TASK["TASK"] == "growing":
			return self.get_new_emoji_x_y()
		else:
			raise ValueError(f"Task {cfg.WORLD.TASK['TASK']}, not implemented")


	def cut_x(self, x):
		""" Removing a random sized box from x and setting it to 0 
				Works with single images and batch_sized images """
		
		if len(x.shape) == 3:
			# Expand first dim
			x = x[None,:]
		
		# Copy, to make sure not to destroy x
		x = np.copy(x)
		
		# Define ranges and pick random box size
		box_size_range_x = [x.shape[-3]*0.2, x.shape[-3]*0.8]
		box_size_range_y = [x.shape[-2]*0.2, x.shape[-2]*0.8]
		box_width = np.random.randint(*box_size_range_x, size=x.shape[0])
		box_height = np.random.randint(*box_size_range_y, size=x.shape[0])
		
		# Randomly select the middle coordinates of the box
		box_x_mid = np.random.randint(x.shape[-3]*0.2, x.shape[-3]*0.8, size=x.shape[0])
		box_y_mid = np.random.randint(x.shape[-2]*0.2, x.shape[-2]*0.8, size=x.shape[0])
		
		# Get the upper left corner of the box and check boundaries
		x1 = box_x_mid - (box_width/2).astype(int)
		y1 = box_y_mid - (box_height/2).astype(int)
		x1 = np.amax([np.zeros(x.shape[0]), x1], axis=0)
		y1 = np.amax([np.zeros(x.shape[0]), y1], axis=0)
		
		# Get lower right corner of the box and check boundaries
		bound_x = x.shape[-3] * np.ones(x.shape[0])
		bound_y = x.shape[-2] * np.ones(x.shape[0])
		x2 = np.amin([bound_x, x1 + box_width], axis=0)
		y2 = np.amin([bound_y, y1 + box_height], axis=0)
		
		# Create Mask to broadcast into batch size
		r1 = np.arange(x.shape[-3])
		r2 = np.arange(x.shape[-2])

		# Broadcasting magic
		mask_x = np.repeat(((x1[:,None] <= r1) & (x2[:,None] >= r1))[:,:,None], x.shape[2], 2)
		mask_y = np.repeat(((y1[:,None] <= r2) & (y2[:,None] >= r2))[:,None,:], x.shape[1], 1)
		mask = mask_x & mask_y

		# Apply mask
		x[mask] = 0.
		return x

	def get_new_emoji_x_y(self):
		""" 
		Get new x and y. For simple config, always init seed and same y 
		Here our x_train is just the single initial seed, y_train is just the
		single emoji goal
		"""
		# TODO new minst/emoji likely can be fully merged
		if cfg.TRAIN.USE_PATTERN_POOL:

			# TODO currently these idx will be replaced with inits from init seed/
			# cuts and then put into the original pool
			self.batch_idx = np.random.choice(len(self.pool_t), cfg.TRAIN.BATCH_SIZE, replace=False)
			
			# Copy not needed! (Tested!)
			x0 = self.pool_t[self.batch_idx]
			y0 = self.y_train[0:cfg.TRAIN.BATCH_SIZE]

			# Based on number of steps set amount to pure seeds
			x0[0:self.seed_idx.numpy()] = self.x_train[0:self.seed_idx.numpy()]

			if cfg.TRAIN.USE_Y_POOL:
				# TODO choose which intervall size? -> with just //4 it could be too big...
				# intervall_size = cfg.TRAIN.BATCH_SIZE - self.seed_idx.numpy()
				intervall_size = cfg.TRAIN.BATCH_SIZE // 4

				expanded_y = np.zeros_like(x0[0:intervall_size])
				# Todo auslagern, use model to access these values
				expanded_y[:,:,:,0:4] = self.y_train[0:intervall_size]
				x0[-intervall_size:] = expanded_y

			if cfg.TRAIN.MUTATE_POOL:
				# 1/4th destroyed pictures
				q_bs = cfg.TRAIN.BATCH_SIZE // 4

				# Randomly removes part of the image for the last 1/4th elements
				x0[-q_bs:] = self.cut_x(x0[-q_bs:])
			# No else path needed as pool values already assigned
		else:
			# Initial most simple configuration -> Fill all of x0 with new x_train imgs.
			x0 = self.x_train[0:cfg.TRAIN.BATCH_SIZE]
			y0 = self.y_train[0:cfg.TRAIN.BATCH_SIZE]

		# self.pool_log.append(np.amax(x0)**2)
		return x0, y0


	def get_new_mnist_x_y(self):
		# Pattern Pool depends on the Model type (starting with very simple (False, False) Model) 
		if cfg.TRAIN.USE_PATTERN_POOL:
			# Use the same pool for training, but replace 1/2 of the images with random images
			# from the training set. First Fourth normal random, last fourth maybe mutated in a weird
			# way.
			self.batch = self.pool.sample(cfg.TRAIN.BATCH_SIZE)
			x0 = np.copy(self.batch.x)
			y0 = self.batch.y

			# we want half of them new. We remove 1/4 from the top and 1/4 from the
			# bottom.
			q_bs = cfg.TRAIN.BATCH_SIZE // 4

			# Initialize the first q_bs images with random x images
			new_idx = self.get_random_train_idx(size=q_bs)
			x0[:q_bs] = self.ca.initialize(self.x_train[new_idx])
			y0[:q_bs] = self.y_train[new_idx]

			# Again, but x not initialized
			new_idx = self.get_random_train_idx(size=q_bs)
			new_x, new_y = self.x_train[new_idx], self.y_train[new_idx]

			if cfg.TRAIN.MUTATE_POOL:
				# Mask with x to 0/1 via booleans 1 if > 0.1
				new_x = tf.reshape(new_x, [q_bs, 28, 28, 1])
				mutate_mask = tf.cast(new_x > 0.1, tf.float32)

				# Mutating x by using a random other images as a mask to throw away some
				# of the hidden states. But only where the number is 0, the other number...
				# TODO look closer into this! And show Images to make sure this is what happens...
				# They mutated -> new image mask is the new image Switching out the numbers
				# Also iln first iteration all will be 0s anyways
				mutated_x = tf.concat([new_x, x0[-q_bs:,:,:,1:] * mutate_mask], -1)

				# - does the last 1/4th elements
				x0[-q_bs:] = mutated_x
				y0[-q_bs:] = new_y
			else:
				x0[-q_bs:] = self.ca.initialize(new_x)
				y0[-q_bs:] = new_y
		else:
			# Initial most simple configuration -> Fill all of x0 with new x_train imgs.
			b_idx = self.get_random_train_idx(size=cfg.TRAIN.BATCH_SIZE)
			x0 = self.ca.initialize(self.x_train[b_idx])
			y0 = self.y_train[b_idx]
		return x0, y0

	@utils.timeit
	def full_train_step(self, current_step):
		""" Run a full training step -> new_imgs, run/update ca """

		# Get current x0, y0 dependent on the model configs/task
		x0, y0 = self.get_new_x_y()

		# TODO get rid of not needed np copys
		# This is needde as x and y are tf.Variables, does not actually faster sadly
		self.x.assign(x0)
		self.y.assign(y0)


		self.num_ca_steps.assign(self.get_num_ca_steps())

		x_out, losses, grads, log = self.apply_grads(self.x, self.y)
		# x, loss = self.update_step(x, y0, seed_idx)

		# self.x_log.append(np.amax(x.numpy())**2)
		self.post_train_step(x_out, y0, current_step)

		self.loss_log.append([x.numpy() for x in losses])
		# [print(grad.numpy().flatten().shape) for grad in grads]
		# max_grad = np.amax(np.hstack([grad.numpy().flatten() for grad in grads]))
		# self.grad_log.append(max_grad)
		# Not sure about returning here or saving locally
		return x0, x_out, losses, grads, log

	@utils.timeit
	def visualize(self, x0, x, loss, grads, run_id=0):
		""" Check current step and visualize current results """
		step_i = len(self.loss_log)

		# Clearing display, to "update" the graph/msg
		
		# Viz loss and results
		if step_i%50 == 0:
			# TODO not working together, with plot as it keeps refreshing
			# And I am unable to see anything
			# disp.visualize_batch(self.ca, x0, x, step_i)
			disp.clear()

			disp.visualize_batch(self.ca, x0, x, step_i)
			# disp.plot_loss(self.x_log, title="max(x)**2")
			# disp.plot_loss(self.pool_log, title="max(Pool)**2")
			# disp.plot_loss(self.grad_log)
			disp.plot_losses(self.loss_log)
			# pass

		# Viz current pool
		if cfg.TRAIN.USE_PATTERN_POOL and step_i%100 == 0:
			pass
			# TODO
			# utils.generate_pool_figures(self.ca, self.pool, step_i)
		
		# Save model
		if step_i%10000 == 0:
			self.save_model()

		# Simple print to show progress, will be overwritten in each step
		print('\r r: %d step: %d, full_loss: %.3f'%(run_id, step_i, loss[0]), end='')
		
		# 
		# disp.show(str('step: %d, log10(loss): %.3f'%(step_i, np.log10(loss))))
		# print(step_i)

	def scatter_simple(self, title=utils.get_cfg_infos()):
		disp.vis_scatter(self.vis, np.array(self.loss_log), title)

	def get_num_ca_steps(self):
		""" Get amount of ca steps to take this iteration based on warmup and cfg"""
		# TODO randomize over batch? 

		cur_step = len(self.loss_log)
		warm_up_steps = cfg.TRAIN.WARM_UP * cfg.TRAIN.NUM_TRAIN_STEPS
		# Check if passed the warmup stage:
		if cur_step >= warm_up_steps:
			return np.random.randint(*cfg.WORLD.CA_STEP_RANGE)
		else:
			cur_ratio = (cur_step * 0.95) / warm_up_steps + 0.05 
			cur_step_range = [int(step * cur_ratio) for step in cfg.WORLD.CA_STEP_RANGE]
			if cur_step_range[0] == cur_step_range[1]:
				return cur_step_range[0]
			else:
				return np.random.randint(*cur_step_range)


	def visdom_loss(self):
		fig = disp.plot_loss(self.loss_log, return_plot=True)
		disp.visdom_plotly(self.vis, fig)

	def reset(self):
		""" Resseting current progress, resseting ca, etc. """
		tf.keras.backend.clear_session()
		self.loss_log = []
		# self.__init__(self.x_train, self.y_train, self.ca, self.vis)

	def get_seed_idx(self):
		""" Updates current seed to pool ratio and current idx"""

		if cfg.TRAIN.FIXED_SEED_RATIO:
			seed_ratio = cfg.TRAIN.FIXED_SEED_RATIO
		else:
			# Update seed ratio
			seed_ratio = 1.

			if len(self.loss_log) > 100:
				seed_ratio = 0.9
			if len(self.loss_log) > 300: # old: 500
				seed_ratio = 0.5
			if len(self.loss_log) > 500: # old: 1000
				# to leave 1 sample in the batch as init seed:
				seed_ratio = 1/cfg.TRAIN.BATCH_SIZE

		assert cfg.TRAIN.BATCH_SIZE * seed_ratio >= 1, "too small seed ratio"

		return int(np.around(cfg.TRAIN.BATCH_SIZE * seed_ratio))

	def post_train_step(self, x, y0, current_step):
		# TODO more complicated for mnist

		# Update seed ratio
		self.seed_idx.assign(self.get_seed_idx())

		# Force pool values to be inside range (-1,1)
		if cfg.TRAIN.POOL_TANH:
			x = tf.math.tanh(x)
		self.pool_t[self.batch_idx] = x.numpy()

		# Save update to tensorboard
		self.ca.tb_log_weights(current_step)


		# if cfg.USE_PATTERN_POOL:
		# 			self.pool.update()
		# 	self.batch.x[:] = x
		# 	self.batch.y[:] = y0 # Not needed in growing, but for classificiation
		# 	self.batch.commit()

	# @tf.function
	def individual_l2_loss(self, x, y):
		""" Creates loss, fitting to task """
		t = y - self.ca.classify(x)
		return tf.reduce_sum(t**2, [1, 2, 3]) / 2

	# @tf.function
	def batch_l2_loss(self, x, y):
		return tf.reduce_mean(self.individual_l2_loss(x, y))

	# I think they get automatically converted if possible, thus decorator does nothing
	def get_losses(self, x, y):
		""" Returns configured loss for model, if grad is true only the total loss 
			loss at [0] is always the total loss used for gradients """
		
		loss = []
		if cfg.TRAIN.LOSS_TYPE == "l2":
			seed_loss = self.batch_l2_loss(x[:self.seed_idx], y[:self.seed_idx])
			if self.seed_idx < cfg.TRAIN.BATCH_SIZE:
				pool_loss = self.batch_l2_loss(x[self.seed_idx:], y[self.seed_idx:])
			else:
				pool_loss = float('NaN')
			full_loss = self.batch_l2_loss(x,y)
			loss = [full_loss, seed_loss, pool_loss]

		return loss	


	def get_apply_grad_fn(self):
		""" Wrapper for apply grad, needed to run several times without resetting """

		@tf.function
		def apply_grad(x, y):

			
			log = {
			"cnn":{
				"max":[],
				"min":[],
				"mean":[]},
			"in":{
				"max":[],
				"min":[],
				"mean":[]},
			"out":{
				"max":[],
				"min":[],
				"mean":[]}
			}

			with tf.GradientTape() as g:
				# TODO tf range is actually slower than writing it out...
				for i in tf.range(self.num_ca_steps):
					if cfg.EXTRA.LOG_LAYERS:
						log["in"]["max"].append(np.amax(x.numpy()))
						log["in"]["min"].append(np.amin(x.numpy()))
						log["in"]["mean"].append(np.mean(x.numpy()))

					out = self.ca(x)
					x, cnn_l = out[0], out[1]

					if cfg.EXTRA.LOG_LAYERS:
						log["out"]["max"].append(np.amax(x.numpy()))
						log["out"]["min"].append(np.amin(x.numpy()))
						log["out"]["mean"].append(np.mean(x.numpy()))

						log["cnn"]["max"].append(np.amax(cnn_l.numpy()))
						log["cnn"]["min"].append(np.amin(cnn_l.numpy()))
						log["cnn"]["mean"].append(np.mean(cnn_l.numpy()))
					
				# TODO is there a better way?
				losses = self.get_losses(x, y)

			grads = g.gradient(losses[0], self.ca.trainable_variables)

			# LAYER_NORM will normalize over a single layer + batch, similar effect
			# As weigth normalization after applying the update
			# TODO if Update gate is true, only a single value will be normed to {-1,1}
			if cfg.TRAIN.LAYER_NORM:
				normed_grads = []
				for layer_grad in grads:
					normed_grads.append(layer_grad / (tf.norm(layer_grad) + 1e-8))

				grads = normed_grads
				# Old implementation
				# grads = [g/(tf.norm(g)+1e-8) for g in grads]
			self.optimizer.apply_gradients(zip(grads, self.ca.trainable_variables))
			return x, losses, grads, log
		return apply_grad


# TODO not used but in original code, not sure for what...
"""
def batch_ce_loss(x, y):
	one_hot = tf.argmax(y, axis=-1)
	# It's ok even if the loss is computed on "dead" cells. Anyway they shouldn't
	# get any gradient propagated through there.
	return tf.compat.v1.losses.sparse_softmax_cross_entropy(one_hot, x)


# assert cfg.LOSS_TYPE in ["l2", "ce"]
# loss_fn = batch_l2_loss if cfg.LOSS_TYPE == "l2" else batch_ce_loss
"""