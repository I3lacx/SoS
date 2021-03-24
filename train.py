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
import os


def get_trainer():
	if cfg.MODEL.NAME == "GANCA":
		return GANCATrainer
	elif cfg.MODEL.NAME == "NCA":
		return NCATrainer
	else:
		raise ValueError(f"Name {cfg.MODEL.NAME} not found/ not implemneted")

class GANCATrainer:
	"""
	Trainer for the GANCA model, can possibly use an abstract Trainer class to ebrt von
	"""

	# TODO last two arguments unused
	def __init__(self, train_set, val_set, model, vis=None, data_seperator=None):
		self.train_set = train_set
		self.val_set = val_set
		self.g_model = model[1].generator
		self.d_model = model[0]
		self.ganca = model[1]

		self.half_batch_size = cfg.TRAIN.BATCH_SIZE // 2

		self.loss_fct = tf.keras.losses.BinaryCrossentropy(label_smoothing=cfg.TRAIN.LABEL_SMOOTHING)
		# self.l2_loss = tf.keras.losses.MeanSquaredError(tf.keras.losses.Reduction.SUM)
		self.l2_loss = self.batch_l2_loss
		self.disc_optimizer = tf.keras.optimizers.Adam(lr=cfg.MODEL.DISC_LR, beta_1=0.5)
		self.ganca_optimizer = tf.keras.optimizers.Adam(lr=cfg.MODEL.GANCA_LR, beta_1=0.5)
		self.gen_optimizer = tf.keras.optimizers.Adam(lr=cfg.MODEL.GEN_L2_LR)
		self.losses = {}
		self.init_loss_dict()
		self.x_old = None
		self.current_step = 0

		self.apply_grad_disc = self.get_apply_grad_fn()
		self.apply_grad_ganca = self.get_apply_grad_fn()
		self.apply_grad_gen = self.get_apply_grad_fn()
		self.init_models()

	def individual_l2_loss(self, x, y):
		""" Creates loss, fitting to task """
		
		t = y - x

		if cfg.TRAIN.REGULARIZER:
			reg = self.regularizer(x[:,:,:,4:])
			return tf.reduce_sum(t**2, [1, 2, 3]) / 2 + reg
		else:
			return tf.reduce_sum(t**2, [1, 2, 3]) / 2

	# @tf.function
	def batch_l2_loss(self, y, x):
		return tf.reduce_mean(self.individual_l2_loss(x, y))


	def init_models(self):
		""" initialize models withtheir training hyperparameters """
		# TODO define somewhere over settings?
		# d_opt = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5)
		# self.d_model.compile(loss='binary_crossentropy', optimizer=d_opt)

		# g_opt = tf.keras.optimizers.Adam(lr=cfg.MODEL.GANCA_LR, beta_1=0.5)
		# self.ganca.compile(loss='binary_crossentropy', optimizer=g_opt)
		pass

	def get_train_batch(self):
		""" get full batch for single training step """
		ganca_in, ganca_y = self.get_input_samples(self.train_set["x"], cfg.TRAIN.BATCH_SIZE, use_old=cfg.TRAIN.GANCA_USE_OLD)
		real_imgs, y_real = self.get_real_samples(self.train_set["y"], self.half_batch_size, noise=cfg.DATA.DISC_INPUT_NOISE)
		fake_imgs, y_fake = self.get_fake_samples(ganca_in, self.half_batch_size, noise=cfg.DATA.DISC_INPUT_NOISE)
		gen_x, gen_y = self.get_gen_samples(cfg.TRAIN.BATCH_SIZE)

		return [ganca_in, ganca_y], [real_imgs, y_real], [fake_imgs, y_fake], [gen_x, gen_y]

	def init_loss_dict(self):
		self.losses["train"] = {}
		self.losses["train"]["disc_loss_real"] = []
		self.losses["train"]["disc_loss_fake"] = []
		self.losses["train"]["ganca_loss"] = []
		self.losses["train"]["gen_l2_loss"] = []

	def update_train_losses(self, d_loss_real, d_loss_fake, ganca_loss, gen_loss):
		self.losses["train"]["disc_loss_real"].append(d_loss_real)
		self.losses["train"]["disc_loss_fake"].append(d_loss_fake)
		self.losses["train"]["ganca_loss"].append(ganca_loss)
		self.losses["train"]["gen_l2_loss"].append(gen_loss)

	# TODO remove my step, make both trainers the same
	# @tf.function
	def full_train_step(self, my_step):
		ganca, real, fake, gen = self.get_train_batch()
		# TODO hardcoded shinenegans
		if False:
			pass
		else:
			_, d_loss_real, disc_grads  = self.apply_grad_disc(self.d_model, *real, self.disc_optimizer)
			_, d_loss_fake, _  = self.apply_grad_disc(self.d_model, *fake, self.disc_optimizer)
			x_old, ganca_loss, _  = self.apply_grad_ganca(self.ganca, *ganca, self.ganca_optimizer, return_x=True)
			if cfg.TRAIN.GEN_L2_LOSS:
				_, gen_l2_loss, _ = self.apply_grad_gen(self.ganca.generator, *gen, self.gen_optimizer, l2_loss=True)
			else:
				gen_l2_loss = 0
			self.x_old = x_old

		self.update_train_losses(d_loss_real, d_loss_fake, ganca_loss, gen_l2_loss)
		losses = [ganca_loss, d_loss_real, d_loss_fake, gen_l2_loss]
		print(f"\r {self.current_step}: disc:(r:{d_loss_real:.2f}, f:{d_loss_fake:.2f}), gan:{ganca_loss:.2f}, gen_l2:{gen_l2_loss:.2f}", end="")
		self.current_step += 1

		return ganca[0], x_old, losses, None, None

	# TODO kwarg is not nice solution..
	def visualize(self, x0, x, losses, *kwargs):
		""" visualizes current trainnig step results """
		if self.current_step % 50 == 0:
			disp.clear()
			# disp.visualize_batch(self.ganca.ca, x0, x, self.current_step, white_background=False)
			disp.plot_ganca_loss(self.losses)

	def add_noise(self, x0):
		noise = tf.random.normal(shape=x0.shape, mean=0, stddev=0.1, dtype=cfg.MODEL.FLOATX)
		x0 = x0 + noise
		return x0

	def get_gen_samples(self, size):
		""" Generator sample supervised training with l2 loss"""
		data_set_size = self.train_set["x"].shape[0]
		random_idxs = np.random.choice(np.arange(data_set_size), size, replace=False)

		x = self.train_set["x"][random_idxs]
		y = self.train_set["y"][random_idxs]

		return x, y

	def get_real_samples(self, data_set, size, noise=False):
		random_idxs = np.random.choice(np.arange(data_set.shape[0]), size, replace=False)
		y_gan = np.ones((size, 1), dtype=cfg.MODEL.FLOATX)

		x_sample = data_set[random_idxs]

		if cfg.DATA.GANCA_NORM_INPUT:
			x_sample = (x_sample - 0.5)/ 0.5

		if noise:
			x_sample = self.add_noise(x_sample)
			if cfg.DATA.GANCA_NORM_INPUT:
				x_sample = np.clip(x_sample, -1, 1)
			else: 
				x_sample = np.clip(x_sample, -1, 1)

		return x_sample, y_gan

	def get_input_samples(self, data_set, size, noise=False, use_old=False):
		# TODO not fast, not optimal way of getting samples...
		# Use old reuses part of saved results for persistency loss

		if cfg.MODEL.LATENT_DIM is not None:
			x0 = tf.random.normal(shape=(size, cfg.MODEL.LATENT_DIM))
			y_gan = np.ones((size, 1), dtype=cfg.MODEL.FLOATX)
			return x0, y_gan

		num_replaced_images = size

		if use_old and self.x_old is not None:
			idx_mask = np.random.choice([True, False], size=cfg.TRAIN.BATCH_SIZE, 
					p=[cfg.TRAIN.KEEP_OUTPUT, 1 - cfg.TRAIN.KEEP_OUTPUT])
			not_idx_mask = np.logical_not(idx_mask)
			num_replaced_images = sum(not_idx_mask)
			# rand_batch_idx = np.random.choice(data_set.shape[0], num_replaced_images, replace=False)

		random_idxs = np.random.choice(np.arange(data_set.shape[0]), num_replaced_images, replace=False)
		y_gan = np.ones((size, 1), dtype=cfg.MODEL.FLOATX)

		x_sample = np.copy(data_set[random_idxs])
		# Normalize between -1/1

		if use_old and self.x_old is not None:
			# x0 = np.zeros((cfg.TRAIN.BATCH_SIZE, ) + self.ganca.input_x_shape, dtype=cfg.MODEL.FLOATX)
			x0 = np.array(self.x_old)
			x0[not_idx_mask] = x_sample
		else:
			x0 = x_sample

		if cfg.DATA.GANCA_NORM_INPUT:
			x0 = (x0 - 0.5)/ 0.5

		if noise:
			x0 = self.add_noise(x0)
			if cfg.DATA.GANCA_NORM_INPUT:
				x0 = np.clip(x0, -1, 1)
			else: 
				x0 = np.clip(x0, -1, 1)

		return x0, y_gan

	def get_fake_samples(self, ganca_in, size, noise=False, validation=False):
		# TODO this is not super nice, but would need to get rid of keras if not like this likely

		gen_out = self.g_model(ganca_in)
		
		fake_img = self.ganca.prepare_output(gen_out)

		if not validation:
			# self.x_old = np.array(fake_img, dtype=cfg.MODEL.FLOATX)
			# Half size to fit for half batch size, not computationaly optimal..
			fake_img = fake_img[::2]

		if noise:
			fake_img = self.add_noise(fake_img)
			# TODO not sure about this
			if cfg.DATA.GANCA_NORM_INPUT:
				fake_img = np.clip(fake_img, -1, 1)
			else: 
				fake_img = np.clip(fake_img, -1, 1)

		# Normalize between -1/1
		# gen_out = gen_out - 0.5 / 0.5
		y_fake = np.zeros((size, 1), dtype=cfg.MODEL.FLOATX)
		return fake_img, y_fake

	def load_weights(self, model_name, path):
		if model_name == "ca":
			# Load weights of the ca model of the generator
			self.ganca.ca.load_weights(path)

	def save_weights(self, path="", name="ca"):
		if path == "":
			path = utils.get_full_log_path()

		utils.ensure_dir(path)

		if not os.path.exists(path + name + ".index"):
			self.ganca.ca.save_weights(path + name)
		else:
			raise ValueError(f"Failed to save because path: {path+name} already exists")


	def get_apply_grad_fn(self):
		""" Wrapper for apply grad, needed to run several times without resetting 
		This is the ganca version!"""

		@tf.function
		def apply_grad(model, x, y, optimizer, return_x=False, l2_loss=False):
			with tf.GradientTape() as g:
				if return_x:
					# Used for ganca model to include fake image in output
					out, fake_img = model(x)
				else:
					out = model(x)
					fake_img = None

				if l2_loss:
					# TODO this cut should be in the output of the gen? But brings other problems..
					loss = self.l2_loss(y, out[:,:,:,:4])
				else:
					loss = self.loss_fct(y, out)

			if return_x:
				# at this point just a way of saying it is the ganca model
				grads = g.gradient(loss, model.generator.trainable_variables)
				if cfg.TRAIN.GANCA_LAYER_NORM:
					grads = [g/(tf.norm(g)+1e-8) for g in grads]
				optimizer.apply_gradients(zip(grads, model.generator.trainable_variables))
			elif l2_loss:
				grads = g.gradient(loss, model.trainable_variables)
				if cfg.TRAIN.GANCA_LAYER_NORM:
					grads = [g/(tf.norm(g)+1e-8) for g in grads]
				optimizer.apply_gradients(zip(grads, model.trainable_variables))
			else:
				grads = g.gradient(loss, model.trainable_variables)
				optimizer.apply_gradients(zip(grads, model.trainable_variables))
			return fake_img, loss, grads
		return apply_grad


class NCATrainer:
	""" 
	Trainer class to handle most of the training stuff, using the ca and the dataset
	cfg should configure loss function, batch size and model configs.
	Could be set in here, but so far in config file.
	"""

	def __init__(self, train_set, val_set, ca, vis, seperator):
		self.train_set = train_set
		self.val_set = val_set
		self.ca = ca 		# cellular automata (model) for computation
		self.vis = vis 		# visdom for visualization
		self.seperator = seperator

		# mostly to set the shape, values will be overwirtten anyways
		# TODO Variable does not really help, just makes it more complicated
		self.x = tf.Variable(self.train_set["x"][0:cfg.TRAIN.BATCH_SIZE], trainable=False)
		self.y = tf.Variable(self.train_set["y"][0:cfg.TRAIN.BATCH_SIZE], trainable=False)
		# How much initial seeds used inside the pool, decreases after steps

		# Set tf Variables
		self.loss_log = []
		self.val_loss_log = {}
		# TODO only for debugging, maybe not the best spot here, not sure how though
		self.grad_log = []
		self.pool_log = []
		self.x_log = []

		# saving old x values for cfg.train.keep_output
		self.x_old, self.y_old, self.old_idx_array = self.get_train_batch(first_call=True)

		self.seed_idx = tf.Variable(0, dtype=tf.int32, trainable=False)
		self.seed_idx.assign(self.get_seed_idx())

		# To set current number of ca steps
		self.num_ca_steps = tf.Variable(0, dtype=tf.int32, trainable=False)

		# Will be used if USE_PATTERN_POOL is true
		self.batch = None

		self.optimizer = self.get_tf_optimizer()
		
		# Get function to apply gradient
		self.apply_grads = self.get_apply_grad_fn()

		self.regularizer = tf.keras.regularizers.L2(0.1)


	def save_weights(self, path="", name="model"):
		if path == "":
			path = utils.get_full_log_path()

		utils.ensure_dir(path)

		if not os.path.exists(path + name + ".index"):
			self.ca.save_weights(path + name)
		else:
			raise ValueError(f"Failed to save because path: {path+name} already exists")

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
		raise ImplementationError("Deprecated!")
		# TODO this is wrong! It should use the dataset by now, not the pool!
		# Only x pool, as y is alwalys the same
		# TODO pool is completely shit!
		pool = np.repeat(self.x_train, cfg.TRAIN.POOL_SIZE // np.shape(self.x_train)[0] , 0)
		return pool

	def get_random_train_idx(self, size):
		return np.random.randint(0, self.x_train.shape[0]-1, size=size)

	@utils.timeit
	def get_train_batch(self, first_call=False):
		if first_call:
			# Get random batch for first call
			rand_batch_idx = np.random.choice(cfg.DATA.TRAIN_SIZE, cfg.TRAIN.BATCH_SIZE, replace=False)
			x0 = np.copy(self.train_set["x"][rand_batch_idx])
			y0 = np.copy(self.train_set["y"][rand_batch_idx])
			idx_array = self.get_idx_array(None, rand_batch_idx, first_call=True)
			return x0, y0, idx_array

		# Randomly decide which inputs will be reused, which not
		# if idx_mask is true, value will be kept for next train step
		idx_mask = np.random.choice([True, False], size=cfg.TRAIN.BATCH_SIZE, 
					p=[cfg.TRAIN.KEEP_OUTPUT, 1 - cfg.TRAIN.KEEP_OUTPUT])
		not_idx_mask = np.logical_not(idx_mask)

		# replace values to be replaced with new x_train_img
		num_replaced_images = sum(not_idx_mask)
		rand_batch_idx = np.random.choice(cfg.DATA.TRAIN_SIZE, num_replaced_images, replace=False)

		idx_array = self.get_idx_array(idx_mask, rand_batch_idx)

		x0 = self.x_old
		y0 = self.y_old
		self.old_idx_array = idx_array

		x0[not_idx_mask] = np.copy(self.train_set["x"][rand_batch_idx])
		y0[not_idx_mask] = np.copy(self.train_set["y"][rand_batch_idx])

		# Todo what to do with idx_array?
		# should not be needed for pure training?
		return x0, y0

	def get_idx_array(self, idx_mask, rand_batch_idx, first_call=False):
		""" 
		Get Array with ids of current selected emojis and which was used from a previous iteration
		Idx 1->N for which emoji was used, based on dataset seperator, negative for used 
		in previous iteration
		"""
		return None
		if first_call:
			print(self.seperator)
		# print("TODO, use the seperator")
		return None


	def cut_x(self, x):
		raise ImplementationError("Moved to dataset")
		# TODO should be moved to dataset
		""" Removing a random sized box from x and setting it to 0 
				Works with single images and batch_sized images """

	def get_new_emoji_x_y(self):
		""" 
		DEPRECATED, SHOULD BE REPLACED!
		Get new x and y. For simple config, always init seed and same y 
		Here our x_train is just the single initial seed, y_train is just the
		single emoji goal
		"""
		raise ImplementationError("Old")
		# TODO new minst/emoji likely can be fully merged
		if cfg.TRAIN.USE_PATTERN_POOL and not cfg.TRAIN.KEEP_OUTPUT:

			# TODO currently these idx will be replaced with inits from init seed/
			# cuts and then put into the original pool
			self.batch_idx = np.random.choice(len(self.pool), cfg.TRAIN.BATCH_SIZE, replace=False)
			
			# Copy not needed! (Tested!)
			x0 = self.pool[self.batch_idx]
			y0 = self.y_train[self.batch_idx]

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
		elif cfg.TRAIN.KEEP_OUTPUT:
			# has all the idx to keep, thus reverse it to throw the others out
			idx_mask = np.random.choice([True, False], size=cfg.TRAIN.BATCH_SIZE, 
						p=[cfg.TRAIN.KEEP_OUTPUT, 1 - cfg.TRAIN.KEEP_OUTPUT])
			not_idx_mask = np.logical_not(idx_mask)

			try:
				x0 = self.x_old.numpy()
				y0 = self.y_old.numpy()
			except AttributeError:
				# TODO this seems to be a hot fix, somewhere in dataset it should be made an array
				x0 = self.x_old
				y0 = self.y_old

			# replace values to be replaced with new x_train_img
			num_replaced_images = sum(not_idx_mask)
			rand_batch_idx = np.random.choice(cfg.TRAIN.POOL_SIZE, num_replaced_images, replace=False)

			x0[not_idx_mask] = np.copy(self.x_train[rand_batch_idx])
			y0[not_idx_mask] = np.copy(self.y_train[rand_batch_idx])

		else:
			# Initial most simple configuration -> Fill all of x0 with new x_train imgs.
			x0 = self.x_train[0:cfg.TRAIN.BATCH_SIZE]
			y0 = self.y_train[0:cfg.TRAIN.BATCH_SIZE]

		# self.pool_log.append(np.amax(x0)**2)
		return x0, y0

	@utils.timeit
	def full_train_step(self, current_step):
		""" Run a full training step -> new_imgs, run/update ca """

		# Get current x0, y0 dependent on the model configs/task
		x0, y0 = self.get_train_batch()

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

	def get_val_loss(self):
		
		# iterate over the whole validation set, then get average
		# loss after 50,100,150 steps?
		# or mean of range *1, *2, ...

		losses = {}
		# starting with the average of all of the
		for emoji_name, emoji_dict in self.seperator["val"].items():
			for key, value in emoji_dict.items():
				cur_x = self.val_set["x"][value[0]:value[1]]
				cur_y = self.val_set["y"][value[0]:value[1]]

				num_steps = int(np.mean(cfg.WORLD.CA_STEP_RANGE))
				# TODO maybe also loss after 1/2/3 times this range
				x = cur_x
				for i in range(num_steps):
					x = self.ca(x)

				cur_loss = self.batch_l2_loss(x, cur_y)
				# The loss is an array so that later I can append those values!
				losses[emoji_name+ "_" + key] = [cur_loss.numpy()]

		return losses


	@utils.timeit
	def visualize(self, x0, x, losses, grads, run_id=0):
		""" Check current step and visualize current results """
		# TODO losses does not need to be used as all information should be in loss_log
		step_i = len(self.loss_log)

		# Clearing display, to "update" the graph/msg
		
		# Viz loss and results
		if step_i%50 == 0:
			# TODO not working together, with plot as it keeps refreshing
			# And I am unable to see anything
			# disp.visualize_batch(self.ca, x0, x, step_i)
			disp.clear()

			disp.visualize_batch(self.ca, x0, x, step_i, white_background=False)
			# disp.plot_loss(self.x_log, title="max(x)**2")
			# disp.plot_loss(self.pool_log, title="max(Pool)**2")
			# disp.plot_loss(self.grad_log)

			# print(self.val_loss_log)
			disp.plot_train_and_val_loss(self.loss_log, self.val_loss_log, y_range="auto")
			# disp.plot_losses(self.loss_log, y_range=(0,400))
			# pass

		# Viz current pool
		if cfg.TRAIN.USE_PATTERN_POOL and step_i%100 == 0:
			pass
			# TODO
			# utils.generate_pool_figures(self.ca, self.pool, step_i)

		# Simple print to show progress, will be overwritten in each step
		print('\r r: %d step: %d, total_loss: %.3f min: %.3f max: %.3f'%(run_id,
		 step_i, losses[0], np.amin(self.loss_log), np.amax(self.loss_log)), end='')
		
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

	def update_val_loss(self, current_step):
		# TODO bit hacky because current step starts at 1...
		if (current_step-1) % cfg.EXTRA.VAL_LOG_INTERVALL == 0:
			val_loss_dict = self.get_val_loss()

			if self.val_loss_log == {}:
				self.val_loss_log = val_loss_dict
			else:
				# Append new values to dict
				for key, value in val_loss_dict.items():
					self.val_loss_log[key] += value

	def post_train_step(self, x, y0, current_step):
		# TODO more complicated for mnist

		# Update seed ratio
		self.seed_idx.assign(self.get_seed_idx())

		# Force pool values to be inside range (-1,1)
		if cfg.TRAIN.POOL_TANH:
			x = tf.math.tanh(x)

		if cfg.TRAIN.KEEP_OUTPUT: 
			# save old x value
			# TODO this saves old x values including hidden information
			# Why?
			self.x_old = np.array(x)
			self.y_old = y0
		elif cfg.TRAIN.USE_PATTERN_POOL:
			self.pool[self.batch_idx] = x.numpy()

		self.update_val_loss(current_step)

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

		if cfg.TRAIN.REGULARIZER:
			reg = self.regularizer(x[:,:,:,4:])
			return tf.reduce_sum(t**2, [1, 2, 3]) / 2 + reg
		else:
			return tf.reduce_sum(t**2, [1, 2, 3]) / 2

	# @tf.function
	def batch_l2_loss(self, x, y):
		return tf.reduce_mean(self.individual_l2_loss(x, y))

	# I think they get automatically converted if possible, thus decorator does nothing
	def get_losses(self, x, y):
		""" Returns configured loss for model, if grad is true only the total loss 
			loss at [0] is always the total loss used for gradients """
		

		full_loss = self.batch_l2_loss(x,y)
		# returs array, so its easier to add different losses to this equation
		return [full_loss]

		# Deprecated functionallity
		loss = []
		if cfg.TRAIN.LOSS_TYPE == "l2":
			seed_loss = self.batch_l2_loss(x[:self.seed_idx], y[:self.seed_idx])
			if self.seed_idx < cfg.TRAIN.BATCH_SIZE:
				pool_loss = self.batch_l2_loss(x[self.seed_idx:], y[self.seed_idx:])
			else:
				if cfg.MODEL.FLOATX == "float32":
					pool_loss = np.float32('NaN')
				elif cfg.MODEL.FLOATX == "float64":
					pool_loss = np.float64('NaN')
				else:
					raise ValueError("Unknown Value for ", cfg.MODEL.FLOATX)

			full_loss = self.batch_l2_loss(x,y)
			loss = [full_loss, seed_loss, pool_loss]

		return loss	


	def get_apply_grad_fn(self):
		""" Wrapper for apply grad, needed to run several times without resetting """

		@tf.function
		def apply_grad(x, y):

			# only used if cfg.extra.log_layers is active
			

			with tf.GradientTape() as g:
				log = []
				# TODO tf range is actually slower than writing it out...
				for i in tf.range(self.num_ca_steps):

					if cfg.EXTRA.LOG_LAYERS:
						out = self.ca(x)
						x = out[0]

						# TODO could be outsorced
						# TODO could add min max and mean here
						log.append([layer for layer in out[-1]])

						"""
						for layer in out:
							if layer[0] in log:
								log[layer[0]].append(layer[1].numpy())
							else:
								log[layer[0]] = [layer[1].numpy()]
						"""
					else:
						x = self.ca(x)

				# TODO is there a better way?
				losses = self.get_losses(x, y)

			grads = g.gradient(losses[0], self.ca.trainable_variables)

			# LAYER_NORM will normalize over a single layer + batch, similar effect
			# As weigth normalization after applying the update
			# TODO if Update gate is true, only a single value will be normed to {-1,1}
			if cfg.TRAIN.LAYER_NORM:
				# This implementation is only useful if I want to remove {-1,1} updates
				# normed_grads = []
				# for layer_grad in grads:
				#	normed_grads.append(layer_grad / (tf.norm(layer_grad) + 1e-8))

				# grads = normed_grads
				# Old implementation
				grads = [g/(tf.norm(g)+1e-8) for g in grads]
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