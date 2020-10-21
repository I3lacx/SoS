"""
File to have all training functions and such
Also for testing?
"""
# Maybe also add the cfg dict as input to set configurations
import config as cfg
import utils
import datasets

import tensorflow as tf
import numpy as np


class Trainer:
	""" 
	Trainer class to handle most of the training stuff, using the ca and the dataset
	cfg should configure loss function, batch size and model configs.
	Could be set in here, but so far in config file.
	"""

	def __init__(self, x_train, y_train, ca):
		self.x_train = x_train
		self.y_train = y_train
		self.ca = ca
		# Steps to take until update could be in cfgs.
		self.ca_num_steps = 20

		# Will be used if USE_PATTERN_POOL is true
		self.batch = None

		self.trainer = self.get_tf_trainer()
		self.pool = self.get_pool()

	def get_tf_trainer(self):
		# TODO globalize lr?
		lr = 1e-3
		lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
			[30000, 70000], [lr, lr*0.1, lr*0.01])
		trainer = tf.keras.optimizers.Adam(lr_sched)
		return trainer

	def get_pool(self):
		starting_indexes = np.random.randint(0, self.x_train.shape[0]-1, size=cfg.POOL_SIZE)
		# pool is a Class defined in utils (kinda related to dataset)
		pool = datasets.SamplePool(x=self.ca.initialize(self.x_train[starting_indexes]).numpy(),
					 y=self.y_train[starting_indexes])
		return pool

	def get_random_train_idx(self, size):
		return np.random.randint(0, self.x_train.shape[0]-1, size=size)

	def get_new_x_y(self):
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
				# Also in first iteration all will be 0s anyways
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

	@tf.function
	def train_step(self, x, y):
		# DO iter_n steps internally, then update and apply losses.
		with tf.GradientTape() as g:
			for i in tf.range(self.ca_num_steps):
				# Single ca step
				x = self.ca(x)
			# TODO outsource to selectable loss function
			loss = self.batch_l2_loss(x, y)
		grads = g.gradient(loss, self.ca.weights)

		# TODO implement actual batch norm or just remove it?
		# Normalizing all gradients by the batch? But why? To tackle gradient explosion?
		grads = [g/(tf.norm(g)+1e-8) for g in grads]
		self.trainer.apply_gradients(zip(grads, self.ca.weights))
		return x, loss

	# TODO super ugly here...
	def post_train_step(self, x, y0):
		if cfg.USE_PATTERN_POOL:
			self.batch.x[:] = x
			self.batch.y[:] = y0 # This gets reordered, so you need to change it.
			self.batch.commit()

	def individual_l2_loss(self, x, y):
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