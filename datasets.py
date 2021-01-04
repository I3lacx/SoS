"""
Containing all stuff about each dataset
Should later be modified into classes, so that I load 
each Dataset as one class with their respective function
"""
import numpy as np
import tensorflow as tf
import time

import utils
import config as cfg

# For loading emoji, utils should be better here?
import requests
import PIL.Image
import io

import math

# Showing emoji
import matplotlib.pyplot as plt

# Only for samples_str
import json 
# --- MNIST --- (later as Class maybe)

def get_data():
	""" Main Function to cal, will give data dependent on task/configs """
	if cfg.WORLD.TASK["TASK"] == "classify":
		# TODO Need to expand for EMNIST
		return get_MNIST()
	elif cfg.WORLD.TASK["TASK"] == "growing":
		return get_emoji_data()
	else:
		raise ValueError()

def get_MNIST():
	print("Loading MNIST ... ", end="")
	start_time = time.time()

	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train = np.array(x_train / 255.0,).astype(np.float32)
	x_test = np.array(x_test / 255.0,).astype(np.float32)

	y_train = to_ten_dim_label(x_train, y_train)
	y_test = to_ten_dim_label(x_test, y_test)
	print(f"Finished after: {np.round(time.time() - start_time, 4):.4f}")

	return (x_train, y_train), (x_test, y_test)

def to_ten_dim_label(x, y):
	""" Converts y into 28x28 images with 10 depth for prediction	"""
	# Maybe improve performance

	# x shape is [b, r, c]
	# y shape is [b]

	# y_res shape is [b, r, c, 10]
	y_res = np.zeros(list(x.shape) + [10])
	# broadcast y to match x shape:
	y_expanded = np.broadcast_to(y, x.T.shape).T
	y_res[x >= 0.1, y_expanded[x >= 0.1]] = 1.0
	return y_res.astype(np.float32)

def find_different_numbers(x_set, y_set, orientation="vertical"):
	result_y = []
	result_x = []
	for i in range(10):
		for x, y in zip(x_set, y_set):
			# Added amax so I can do it without original y, getting idx
			if np.amax(np.where(y == 1)[2]) == i:
				result_y.append(utils.color_labels(x, y))
				result_x.append(x)
				break
	assert len(result_y) == 10, f"result_y: {result_y}"

	result_y = np.concatenate(result_y, axis=0 if orientation == "vertical" else 1)
	result_x = np.stack(result_x)

	return result_y, result_x


def show_number_legends():
	raise NotImplementedError("Not Tested, but Code should be there")
	numbers_legend, x_legend = find_different_numbers(x_train, y_train, y_train_pic)
	numbers_legend_horiz, _ = find_different_numbers(x_train, y_train, y_train_pic, "horizontal")
	plt.imshow(zoom(numbers_legend_horiz))
	plt.show()
	imshow(zoom(numbers_legend_horiz))


# TODO inside class? If inside class no parameters, as x/y in class
def get_samples_str(x_train, y_train):
	numbers_legend, x_legend = find_different_numbers(x_train, y_train)
	samples_str = json.dumps(x_legend.tolist())
	return samples_str

class SamplePool:
	"""
	Kinda weird Python magic here, but in the end should give me
	.x and .y properties of each samplePool here. So that I can
	just do: pool.sample(batch_size).x to get my x images.
	"""
	def __init__(self, *, _parent=None, _parent_idx=None, **slots):
		self._parent = _parent
		self._parent_idx = _parent_idx
		self._slot_names = slots.keys()
		self._size = None 
		for k, v in slots.items():
			if self._size is None:
				self._size = len(v)
			assert self._size == len(v)
			setattr(self, k, np.asarray(v))

	def sample(self, n):
		""" Sample n random elements from pool and return new pool of size n"""
		# TODO is more tricky for MNIST
		idx = np.random.choice(self._size, n, replace=False)
		return self.pool[idx]
		# batch = {k: getattr(self, k)[idx] for k in self._slot_names}
		# batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
		# return batch

	def commit(self):
		""" I think it updates the parent via the batch or so """
		for k in self._slot_names:
			getattr(self._parent, k)[self._parent_idx] = getattr(self, k)


# Not sure if it should be a class...
class Emoji:
	
	def __init__(self, emoji_str):
		self.emoji_img = self.get_emoji_img(emoji_str)
		# self.emoji_rgb = utils.rgba_to_rgb(self.emoji_img)

	def get_emoji_img(self, emoji_str):
		code = self.get_emoji_char(emoji_str)

		# code = hex(ord(emoji_char))[2:].lower()
		emoji_url = 'https://github.com/googlefonts/noto-emoji/raw/master/png/128/emoji_u%s.png'%code
		emoji_img = self.load_image(emoji_url)
		return emoji_img

	def get_emoji_char(self, emoji_str):
		# TODO add more emojis, maybe put func in utils?
		if emoji_str == "lizard":
			return hex(ord("🦎"))[2:].lower()
		elif emoji_str == "double_exclamation_mark":
			return "203c"
		elif emoji_str == "heavy_check_mark":
			return "2714"
		elif emoji_str == "heart":
			return "2764"
		else:
			raise ValueError(f"string: {emoji_str} not implemented")

	def load_image(self, url, max_size=cfg.WORLD.TASK["SIZE"]):
		r = requests.get(url)
		img = PIL.Image.open(io.BytesIO(r.content))
		img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
		img = np.float32(img)/255.0

		assert np.shape(img)[1] <= cfg.WORLD.TASK["SIZE"], f"loaded emoji too big: {np.shape(img)[1]}"
		return img

	def show_emoji(self):
		plt.imshow(self.emoji_img)
		plt.axis("off")
		plt.show()


def add_padding(target, full_size):
	""" Add padding to img, such that it fits in full_size for rgba img"""
	img = np.zeros((full_size, full_size, np.shape(target)[2]), np.float32)

	width = np.shape(target)[0]
	height = np.shape(target)[1]

	x_0 = full_size//2 - width//2
	x_1 = full_size//2 + math.ceil(width/2)

	y_0 = full_size//2 - height//2
	y_1 = full_size//2 + math.ceil(height/2)

	img[x_0:x_1, y_0:y_1,:] = target

	print(img.shape)
	return img

def get_emoji_data(random=False):
	""" Gives data in the style of classificiation, for consistency """
	# Keep α-Channel to use as target as well
	y = Emoji(cfg.WORLD.TASK["TARGET"]).emoji_img
	
	if random:
		x = random_seed()
	else:
		x = init_seed()

	y = add_padding(y, cfg.WORLD.SIZE)

	x_batch = np.repeat(x, cfg.TRAIN.POOL_SIZE, 0)
	y_batch = np.repeat(np.expand_dims(y, 0), cfg.TRAIN.POOL_SIZE, 0)

	# Test Data not useful, thus just None for consistency (I could use the same x/y)
	return (x_batch, y_batch), (None, None)


# TODO not sure where to fit seeds, but here seems k
def init_seed():
	""" Get initial World seed with single alive cell in the middle """
	assert cfg.MODEL.CHANNEL_N >= 3, "CHANNEL_N has to be greater 3 to account for rgb"
	seed = np.zeros([1, cfg.WORLD.SIZE, cfg.WORLD.SIZE, cfg.MODEL.CHANNEL_N], np.float32)

	# Selecting the middle pixel and setting it to black, no just activating middle cells
	# TODO (why 3:)-> make cells alive and motivate computation
	seed[:, cfg.WORLD.SIZE//2, cfg.WORLD.SIZE//2, 3:] = 1.0


	return seed

def random_seed():
	seed = np.random.uniform(0, 1, [1, cfg.WORLD.SIZE, cfg.WORLD.SIZE, cfg.MODEL.CHANNEL_N])

	# Selecting the middle pixel and setting it to black
	# seed[:, cfg.TARGET_SIZE//2, cfg.TARGET_SIZE//2, :3] = 1.0

	return seed