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

		# How much initial seeds used inside the pool, decreases after steps

		self.seed_ratio = 1
		# Will be used if USE_PATTERN_POOL is true
		self.batch = None

		self.trainer = self.get_tf_trainer()
		self.pool = self.get_pool()

		# TODO remove for testing only
		self.pool_t = np.repeat(self.x_train[0:1], 80, 0)
		print("pool size ", self.pool_t.shape)
		self.loss_log = []

	def get_tf_trainer(self):
		# TODO globalize lr?
		# TODO change lr/optimizer
		lr = cfg.LR
		lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
			[30000, 70000], [lr, lr*0.1, lr*0.01])
		trainer = tf.keras.optimizers.Adam(lr)
		return trainer

	def get_pool(self):
		if cfg.MODEL_TASK == "growing":
			# Use x,y directly as sample pool (always the same)
			# x and y train are created for pool size
			pool = datasets.SamplePool(x=self.x_train, y=self.y_train)
		elif cfg.MODEL_TASK == "classify":
			starting_indexes = np.random.randint(0, self.x_train.shape[0]-1, size=cfg.POOL_SIZE)
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
		if cfg.MODEL_TASK == "classify":
			return self.get_new_mnist_x_y()
		elif cfg.MODEL_TASK == "growing":
			return self.get_new_emoji_x_y()
		else:
			raise ValueError(f"Task {cfg.MODEL_TASK}, not implemented")


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
		if cfg.USE_PATTERN_POOL:

			# TODO currently these idx will be replaced with inits from init seed/
			# cuts and then put into the original pool
			self.batch_idx = np.random.choice(len(self.pool_t), cfg.BATCH_SIZE, replace=False)
			x0 = self.pool_t[self.batch_idx]
			
			# Keep one seed 
			# np copy should not be needed
			# x0[0] = np.copy(self.x_train[0])
			y0 = np.copy(self.y_train[0:cfg.BATCH_SIZE])
			# return x0, y0

			# Sample full Batch of pool samples
			# self.batch = self.pool.sample(cfg.BATCH_SIZE)
			# TODO np.copy needed? -> should not be needed 
			# x0 = np.copy(self.batch.x)
			#y0 = self.batch.y

			# Based on number of steps set amount to pure seeds
			seed_idx = int(np.around(cfg.BATCH_SIZE*self.seed_ratio))
			x0[0:seed_idx] = self.x_train[0:seed_idx]
			
			# return x0, y0
			
			# Initialize the first q_bs images with normal starting seeds,
			# new_idx = self.get_random_train_idx(size=q_bs)
			# x0[:q_bs] = self.x_train[new_idx]
			# Not needed as y never changes
			# y0[:q_bs] = self.y_train[new_idx]

			# Again, but x not initialized
			# new_idx = self.get_random_train_idx(size=q_bs)
			# new_x, new_y = self.x_train[new_idx], self.y_train[new_idx]

			if cfg.MUTATE_POOL:
				# 1/4th destroyed pictures
				q_bs = cfg.BATCH_SIZE // 4

				# Randomly removes part of the image for the last 1/4th elements
				x0[-q_bs:] = self.cut_x(x0[-q_bs:])
			# No else path needed as pool values already assigned
		else:
			# Initial most simple configuration -> Fill all of x0 with new x_train imgs.
			x0 = self.x_train[0:cfg.BATCH_SIZE]
			y0 = self.y_train[0:cfg.BATCH_SIZE]
		return x0, y0


	def get_new_mnist_x_y(self):
		# Pattern Pool depends on the Model type (starting with very simple (False, False) Model) 
		if cfg.USE_PATTERN_POOL:
			# Use the same pool for training, but replace 1/2 of the images with random images
			# from the training set. First Fourth normal random, last fourth maybe mutated in a weird
			# way.
			self.batch = self.pool.sample(cfg.BATCH_SIZE)
			x0 = np.copy(self.batch.x)
			y0 = self.batch.y

			# we want half of them new. We remove 1/4 from the top and 1/4 from the
			# bottom.
			q_bs = cfg.BATCH_SIZE // 4

			# Initialize the first q_bs images with random x images
			new_idx = self.get_random_train_idx(size=q_bs)
			x0[:q_bs] = self.ca.initialize(self.x_train[new_idx])
			y0[:q_bs] = self.y_train[new_idx]

			# Again, but x not initialized
			new_idx = self.get_random_train_idx(size=q_bs)
			new_x, new_y = self.x_train[new_idx], self.y_train[new_idx]

			if cfg.MUTATE_POOL:
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
			b_idx = self.get_random_train_idx(size=cfg.BATCH_SIZE)
			x0 = self.ca.initialize(self.x_train[b_idx])
			y0 = self.y_train[b_idx]
		return x0, y0

	@utils.timeit
	@tf.function
	def update_step(self, x, y):
		# DO iter_n steps internally, then update and apply losses.
		with tf.GradientTape() as g:
			# Selecting a random amount of steps from a list
			# Using a star unpacks the list
			num_ca_steps = np.random.randint(*cfg.CA_STEP_RANGE)
			for i in tf.range(num_ca_steps):
				# Single ca step
				x = self.ca(x)
			# TODO outsource to selectable loss function
			loss = self.batch_l2_loss(x, y)
		grads = g.gradient(loss, self.ca.weights)
		# LAYER_NORM will normalize over a single layer + batch, similar effect
		# As weigth normalization after applying the update
		if cfg.LAYER_NORM:
			grads = [g/(tf.norm(g)+1e-8) for g in grads]
		self.trainer.apply_gradients(zip(grads, self.ca.weights))
		return x, loss

	@utils.timeit
	def full_train_step(self):
		""" Run a full training step -> new_imgs, run/update ca """

		# Get current x0, y0 dependent on the model configs/task
		x0, y0 = self.get_new_x_y()

		# Very much needed! Bad luck or it just won't train as I tested
		x = np.copy(x0)

		# Runs ca for ca_num_steps and applies gradients, returns new x
		x, loss = self.update_step(x, y0)

		# TODO hotfix!
		# self.pool_t[self.batch_idx] = x

		# Not nice, applying updates to the pool if specific model configs are set
		# TODO use? if len(self.loss_log) > 501:
		self.post_train_step(x, y0)

		self.loss_log.append(loss.numpy())
		# Not sure about returning here or saving locally
		return x0, x, loss

	@utils.timeit
	def visualize(self, x0, x, loss):
		""" Check current step and visualize current results """
		step_i = len(self.loss_log)

		# Clearing display, to "update" the graph/msg
		

		# Viz loss and results
		if step_i%50 == 0:
			# TODO not working together, with plot as it keeps refreshing
			# And I am unable to see anything
			# disp.visualize_batch(self.ca, x0, x, step_i)
			disp.clear()
			disp.plot_loss(self.loss_log)
			pass

		# Viz current pool
		if cfg.USE_PATTERN_POOL and step_i%100 == 0:
			pass
			# TODO
			# utils.generate_pool_figures(self.ca, self.pool, step_i)
		
		# Save model
		if step_i%10000 == 0:
			self.save_model()

		# Simple print to show progress, will be overwritten in each step
		print('\r step: %d, log10(loss): %.3f'%(step_i, np.log10(loss)), end='')
		
		# 
		# disp.show(str('step: %d, log10(loss): %.3f'%(step_i, np.log10(loss))))
		# print(step_i)

	def scatter_simple(self, title=utils.get_compact_title_str()):
		disp.vis_scatter(self.vis, np.array(self.loss_log), title)


	def reset(self):
		""" Resseting current progress, resseting ca, etc. """
		self.loss_log = []
		self.ca = models.CAModel()

	# TODO super ugly here...
	def post_train_step(self, x, y0):
		# TODO more complicated for mnist
		# TODO outsource

		# Update seed ratio
		if len(self.loss_log) > 100:
			self.seed_ratio = 0.9
		if len(self.loss_log) > 300: # old: 500
			self.seed_ratio = 0.5
		if len(self.loss_log) > 600: # old: 1000
			# to leave 1 sample in the batch as init seed:
			self.seed_ratio = 1/cfg.BATCH_SIZE

		assert cfg.BATCH_SIZE * self.seed_ratio >= 1, "too small seed ratio"

		self.pool_t[self.batch_idx] = x

		# if cfg.USE_PATTERN_POOL:
		# 			self.pool.update()
		# 	self.batch.x[:] = x
		# 	self.batch.y[:] = y0 # Not needed in growing, but for classificiation
		# 	self.batch.commit()

	def individual_l2_loss(self, x, y):
		""" Creates loss, fitting to task """
		t = y - self.ca.classify(x)
		return tf.reduce_sum(t**2, [1, 2, 3]) / 2

	def batch_l2_loss(self, x, y):
		return tf.reduce_mean(self.individual_l2_loss(x, y))


	# By me:
	def single_run(ca, num_steps, prefix, disable_black=True):
		raise NotImplementedError("Not yet tested after reordering")
		# Initiialize with single random index from train data
		new_idx = np.random.randint(0, x_train.shape[0]-1)
		x = ca.initialize(np.expand_dims(x_train[new_idx,:,:], 0))
		# TODO where 20 come from?
		frames = []
		with utils.VideoWriter(prefix + ".mp4") as vid:
					# tqdm is progress
			for i in tqdm.trange(-1, num_steps):
				if i == -1:
					image = utils.classify_and_show(ca, x, False)
				else:
					x = ca(x)
					image = utils.classify_and_show(ca, x, disable_black)
				# vis_extended = np.concatenate((image, np.ones((86, image.shape[1], 3))), axis=0) 
				im = np.uint8(image*255)
				im = PIL.Image.fromarray(im)
				# not needed
				# im.paste(slider, box=(0, image.shape[0]+20))
				draw = PIL.ImageDraw.Draw(im)
				# p_x = 3+(((image.shape[1]-5-3)/num_steps)*i)
				# draw.rectangle([p_x, image.shape[0]+21, p_x+5, image.shape[0]+42], fill="#434343bd")
			vid.add(np.uint8(im))

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