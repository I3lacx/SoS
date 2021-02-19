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

# for edge detection
import cv2

import math

# Showing emoji
import matplotlib.pyplot as plt

# Only for samples_str
import json 
# --- MNIST --- (later as Class maybe)


def get_dataset_class():
	""" Function to get all data! 
	Dependent on the config file, dataset and configurations will be chosen here. """
	if cfg.DATA.NAME == "EMOJI":
		data_class = EmojiDataset()
	elif cfg.DATA.NAME == "FACES":
		data_class = FacesDataset()
	else:
		raise ValueError()

	return data_class

class Dataset():
	""" abstract ish class from which emoji and faces dataset will inherit, covers 
	functionallity usefull for both  datasets 
	How the dataset functions, will later be decided by the config DATA"""

	def __init__(self):
		# initialize
		pass

	def add_noise(self, images):
		""" Adds noise to input 
		images contains an array of images, e.g. [100, 40, 40, 16]"""
		pass

	def add_distortions(self, images):
		""" Adds distortions to the images 
		images contains an array of images, e.g. [100, 40, 40, 16]"""
		pass

	def create_dataset(self):
		print(f"Creating {self.name} Dataset")

		x_shape = (cfg.DATA.GRID_SIZE, cfg.DATA.GRID_SIZE, cfg.MODEL.CHANNEL_N)
		num_partitions = len(self.targets)

		train_x, train_y, train_sep = self.create_partitions((cfg.DATA.TRAIN_SIZE,) + x_shape,
															 self.fill, num_partitions)
		val_x, val_y, val_sep = self.create_partitions((cfg.DATA.VAL_SIZE,) + x_shape,
															 self.fill, num_partitions)
		test_x, test_y,test_sep = self.create_partitions((cfg.DATA.TEST_SIZE,) + x_shape,
															 self.fill, num_partitions)

		dataset = {
			"train":
				{"x": train_x,
				 "y": train_y},
			"val":
				{"x": val_x,
				 "y": val_y},
			"test":
				{"x": test_x,
				 "y": test_y}
		}

		seperator = {
			"train": train_sep,
			"val": val_sep,
			"test": test_sep
		}

		assert train_x.dtype == np.dtype(cfg.MODEL.FLOATX), "Wrong dtype in train_x"

		return dataset, seperator
		

class EmojiDataset(Dataset):
	""" Overwriting Dataset, to add specific emoji functionaity to load and prepare this data """
	name = "Emoji"

	def __init__(self):
		# TODO wow defined here as self, but using the cfg. inforamtion in the class, not good
		self.targets = self.get_targets()

		# TODO targets for train/val seperate 
		# Defines the boundaries where which type of data is stored
		self.seperator = {}
		self.dataset, self.seperator = self.create_dataset()

	def get_targets(self):
		""" if cfg.DATA.TARGETS is a list/tuple, will use this,
		if its a number will return a number of randomly selected emojis """

		if type(cfg.DATA.TARGETS) in ["list", "tuple"]:
			targets = cfg.DATA.TARGETS
		elif type(cfg.DATA.TARGETS) == int:
			idx_list = np.random.choice(range(len(emoji_dict)), size=cfg.DATA.TARGETS, replace=False)
			emoji_dict_list = list(emoji_dict)
			targets = tuple([emoji_dict_list[idx] for idx in idx_list])
			print("Created Target list:", targets)

		return targets

	def get_set_seperator(self, sep, cur_idx):
		""" Turn emoji seperator into part of a set seperator """
		for key, value in sep.items():
			# Update the list by adding the idx to both elements
			sep.update({key: [el + cur_idx for el in value]})
		return sep


	def create_partitions(self, full_size, fill_function, num_partitions):
		""" creates x partion of size full_size: [set_size, width, height, depth]
		using the fill function, which accepts a size and returns tha same shape as full_size[1:]"""
		
		x_partition = np.empty(full_size, dtype=cfg.MODEL.FLOATX)
		y_partition = np.empty(full_size[:3] + (4,), dtype=cfg.MODEL.FLOATX)

		set_seperator = {}

		size_all = full_size[0] // num_partitions
		size_extra = full_size[0] % num_partitions
		cur_idx = 0

		for i in range(num_partitions):
			# To evenly distribute the extra spaces
			if size_extra > 0:
				cur_size = size_all + 1
				size_extra -= 1
			else:
				cur_size = size_all

			# if always the same fill function, just call it here...
			cur_x, cur_y, emoji_sep = fill_function((cur_size,) + full_size[1:], i)

			set_seperator[self.targets[i]] = self.get_set_seperator(emoji_sep, cur_idx)
			x_partition[cur_idx:cur_idx + cur_size] = cur_x
			y_partition[cur_idx:cur_idx + cur_size] = cur_y

			cur_idx += cur_size

		assert cur_idx == full_size[0], "Not every value in empty has been filled!"
		assert not np.any(np.abs(x_partition) > 1e10), "A abs value is higher than 1e10, likely not filled!"

		return x_partition, y_partition, set_seperator

	def fill(self, size, idx):
		""" main fill function for x, changes how to fill it dependent on the config
		returns correctly sized x and y """

		emoji_obj = Emoji(self.targets[idx])

		emoji_y = emoji_obj.emoji_img
		single_y = add_padding(emoji_y, cfg.DATA.GRID_SIZE)[None, :]
		full_y = np.repeat(single_y, size[0], axis=0)

		damaged_idx = int(size[0] * cfg.DATA.DAMAGE)

		if cfg.DATA.SEED == "EDGES":
			# Create edge image
			edge_emoji = emoji_obj.get_gray_edge_image()
			edge_emoji = add_padding(edge_emoji, cfg.DATA.GRID_SIZE)

			single_x = np.zeros((1,) + size[1:])
			single_x[:,:,:,:4] = edge_emoji[None, :]

			full_x = np.repeat(single_x, size[0], axis=0)	
 
		if cfg.DATA.NOISE > 0:

			# Add gaussian noise to all images
			noise = np.random.normal(0, cfg.DATA.NOISE, size)

			if cfg.DATA.ONLY_POS_NOISE:
				noise = np.abs(noise)

			# apply noise	
			full_x = full_x + noise

			if cfg.DATA.CLIP_NOISE:
				full_x = np.clip(full_x, 0, 1)
			
		if cfg.DATA.DAMAGE > 0:
			# Remove random rectangles from image
			full_x[0:damaged_idx] = self.cut_x(full_x[0:damaged_idx])

		cur_seperator = self.get_seperator_per_emoji(size[0], damaged_idx)

		return full_x, full_y, cur_seperator

	def get_seperator_per_emoji(self, size, damaged_idx):
		# Returns the seperator for a single emoji, last idx not inclusive
		seperator = {}

		if damaged_idx > 0:
			seperator["DAMAGE"] = [0, damaged_idx]

		# fancy one liner, but will stick to readable code...
		# seperator[cfg.DATA.SEED + "_NOISE" if cfg.DATA.NOISE else ""] = [damaged_idx, size]

		if cfg.DATA.NOISE:
			seperator[cfg.DATA.SEED + "_NOISE"] = [damaged_idx, size]
		else:
			seperator[cfg.DATA.SEED] = [damaged_idx, size]

		return seperator

	def cut_x(self, x):
		# TODO should be moved to dataset
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

	def get_all_img_types(self, data_set, idx_offset, random_idx=False):
		""" Returns dict with imgs for each type of data
		offset 0 to alwalys get the first, random to get a random in correct range"""
		img_dict = {}
		for emoji_name, emoji_dict in self.seperator[data_set].items():
			for key, value in emoji_dict.items():
				if random_idx:
					cur_idx = np.random.randint(value[0], value[1])
				else:
					cur_idx = value[0] + idx_offset
				img_dict[emoji_name + "_" + key] = self.dataset[data_set]["x"][cur_idx]
		return img_dict
 

class FacesDataset(Dataset):
	""" Overwriting Dataset, to add specific Faces functionality to load and prepare this data """
	# from: https://github.com/bchao1/Anime-Face-Dataset

	name = "Faces"

	def __init__(self):
		self.targets = self.get_targets()

		# Defines the boundaries where which type of data is stored
		self.seperator = {}
		# create dataset is the super call
		self.dataset, self.seperator = self.create_dataset()

	def get_targets(self):
		# only written for amount of targets, will create file name list for target faces
		assert type(cfg.DATA.TARGETS) == int

		# TODO outsource in utils
		import os

		all_file_names = os.listdir("faces_dataset/cropped/")
		idx_list = np.random.choice(range(len(all_file_names)), size=cfg.DATA.TARGETS, replace=False)
		targets = tuple([all_file_names[i] for i in idx_list])
		return targets

	def create_partitions(self, full_size, fill_function, num_partitions):
		# TODO might be identical to the one from emojis
		""" creates x partion of size full_size: [set_size, width, height, depth]
		using the fill function, which accepts a size and returns tha same shape as full_size[1:]"""
		
		x_partition = np.empty(full_size, dtype=cfg.MODEL.FLOATX)
		y_partition = np.empty(full_size[:3] + (4,), dtype=cfg.MODEL.FLOATX)

		set_seperator = {}

		size_all = full_size[0] // num_partitions
		size_extra = full_size[0] % num_partitions
		cur_idx = 0

		for i in range(num_partitions):
			# To evenly distribute the extra spaces
			if size_extra > 0:
				cur_size = size_all + 1
				size_extra -= 1
			else:
				cur_size = size_all

			# if always the same fill function, just call it here...
			cur_x, cur_y, emoji_sep = fill_function((cur_size,) + full_size[1:], i)

			set_seperator[self.targets[i]] = self.get_set_seperator(emoji_sep, cur_idx)
			x_partition[cur_idx:cur_idx + cur_size] = cur_x
			y_partition[cur_idx:cur_idx + cur_size] = cur_y

			cur_idx += cur_size

		assert cur_idx == full_size[0], "Not every value in empty has been filled!"
		assert not np.any(np.abs(x_partition) > 1e10), "A abs value is higher than 1e10, likely not filled!"

		return x_partition, y_partition, set_seperator

	def get_set_seperator(self, sep, cur_idx):
		# TOOD is emoji thingy
		""" Turn emoji seperator into part of a set seperator """
		for key, value in sep.items():
			# Update the list by adding the idx to both elements
			sep.update({key: [el + cur_idx for el in value]})
		return sep

	def get_face_from_name(self, name, shape):
		# TODO add path in configs somewhere!
		# Todo alpha channel will be only 1s
		img = cv2.imread(f"faces_dataset/cropped/{name}", cv2.IMREAD_COLOR)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
		img = cv2.resize(img, shape)
		return img

	def face_to_edges(self, img):
		gray_img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
		edges = cv2.Canny(gray_img,100,200)
		edges = np.repeat(edges[:,:,None], 4, axis=2)
		edges = edges.astype(cfg.MODEL.FLOATX) / 255.
		return edges


	def fill(self, size, idx):
		# TODO not hard coded shape and so on
		single_y = self.get_face_from_name(self.targets[idx], shape=(64,64))
		single_y_norm = single_y[None, :].astype(cfg.MODEL.FLOATX) / 255.
		full_y = np.repeat(single_y_norm, size[0], axis=0)

		# TODO just simple edges no noise implemented so far
		face_edges = self.face_to_edges(single_y)

		single_x = np.zeros((1,) + size[1:])
		single_x[:,:,:,:4] = face_edges[None, :]
		full_x = np.repeat(single_x, size[0], axis=0)

		cur_seperator = {"EDGES": [0, size[0]]}

		# TODO
		# cur_seperator = self.get_seperator()

		return full_x, full_y, cur_seperator



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
	return y_res.astype(cfg.MODEL.FLOATX)

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
	
	raw_img_sizes = [32, 72, 128, 512]

	def __init__(self, emoji_str):
		self.emoji_str = emoji_str
		self.emoji_img = self.get_emoji_img(emoji_str)
		# self.emoji_rgb = utils.rgba_to_rgb(self.emoji_img)

	def get_emoji_img(self, emoji_str):
		code = self.get_emoji_char(emoji_str)

		# Select the smallest possible size to load from
		size_idx = 0
		while self.raw_img_sizes[size_idx] < cfg.DATA.TARGET_SIZE and cfg.DATA.TARGET_SIZE <= 512:
			size_idx += 1
		
		emoji_url = "https://github.com/googlefonts/noto-emoji/raw/" + \
					f"master/png/{self.raw_img_sizes[size_idx]}/emoji_u{code}.png"

		emoji_img = self.load_image(emoji_url)
		return emoji_img

	def get_emoji_char(self, emoji_str):
		try:
			emoji_str = emoji_dict[emoji_str]
		except KeyError:
			raise ValueError(f"string: {emoji_str} not implemented")

		return emoji_str

	def load_image(self, url, max_size=cfg.DATA.TARGET_SIZE):
		r = requests.get(url)
		img = PIL.Image.open(io.BytesIO(r.content))
		img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)

		# now (15.01.2021) converts image into np array
		img = np.array(img)
		img = img.astype(cfg.MODEL.FLOATX)/255.0

		assert np.shape(img)[1] <= cfg.DATA.TARGET_SIZE, f"loaded emoji too big: {np.shape(img)[1]}"
		return img

	def show_emoji(self):
		plt.imshow(self.emoji_img)
		plt.axis("off")
		plt.show()

	def get_gray_edge_image(self, img=None):
		""" turns input emoji image into edge image """
		# will expect a color image, will return a color image 
		if img is None:
			img = self.emoji_img

		edges = get_gray_edge_image(img)
		return edges

	def save_to_file(self, path=""):
		# saves own image to path as png
		if path == "": 
			path = f"{self.emoji_str}.png"

		utils.save_img_to_file(self.emoji_img, path)


def get_gray_edge_image(img):
	""" turns input image into edge image """
	# will expect a color image, will return a color image 

	img[:,:,:3] *= img[:,:,3:4]
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	img = np.uint8(255 * img)
	edges = cv2.Canny(img,100,200)
	edges = np.repeat(edges[:,:,None], 4, axis=2)
	edges = edges.astype(cfg.MODEL.FLOATX) / 255.
	return edges


def add_padding(target, full_size):
	""" Add padding to img, such that it fits in full_size for rgba img"""
	img = np.zeros((full_size, full_size, np.shape(target)[2]), cfg.MODEL.FLOATX)

	width = np.shape(target)[0]
	height = np.shape(target)[1]

	x_0 = full_size//2 - width//2
	x_1 = full_size//2 + math.ceil(width/2)

	y_0 = full_size//2 - height//2
	y_1 = full_size//2 + math.ceil(height/2)

	img[x_0:x_1, y_0:y_1,:] = target

	return img


# TODO all functions below are not used and replaced by the emoji class!


def get_emoji_data(random=False):
	""" Gives data in the style of classificiation, for consistency """
	# Keep α-Channel to use as target as well
	# TODO list of emojis expected!
	y = Emoji(cfg.DATA.TARGETS).emoji_img
	
	if random:
		x = random_seed()
	else:
		x = init_seed()

	y = add_padding(y, cfg.DATA.GRID_SIZE)

	x_batch = np.repeat(x, cfg.TRAIN.POOL_SIZE, 0)
	y_batch = np.repeat(np.expand_dims(y, 0), cfg.TRAIN.POOL_SIZE, 0)

	# Test Data not useful, thus just None for consistency (I could use the same x/y)
	return (x_batch, y_batch), (None, None)


# TODO use this function? Is now implemented in the fill function...
def get_single_emoji_edge_data(emoji_str="", noise=False, amount=cfg.TRAIN.POOL_SIZE):
	if emoji_str == "":
		# TODO list of emojis expected
		emoji_str = cfg.DATA.TARGETS

	module = Emoji(emoji_str)
	y = module.emoji_img
	x = module.get_gray_edge_image(y)

	# copy values from gray image across all channels
	x = np.repeat(x[:,:,None], cfg.MODEL.CHANNEL_N, axis=2)

	y = add_padding(y, cfg.DATA.GRID_SIZE)
	x = add_padding(x, cfg.DATA.GRID_SIZE)

	x_batch = np.repeat(np.expand_dims(x, 0), amount, 0)
	y_batch = np.repeat(np.expand_dims(y, 0), amount, 0)

	# TODO test data
	return (x_batch, y_batch), (None, None)


def get_multiple_emoji_edge_data(emoji_strings=[], noise=False):
	# TODO this is the easy case to have the same batch but with two different images
	# harder if I add noise... But I could also add the noise differently each time...
	emoji_strings = cfg.DATA.TARGETS

	full_batch_x = np.empty([cfg.TRAIN.POOL_SIZE, cfg.DATA.GRID_SIZE, cfg.DATA.GRID_SIZE, cfg.MODEL.CHANNEL_N],
							dtype=cfg.MODEL.FLOATX)
	full_batch_y = np.empty([cfg.TRAIN.POOL_SIZE, cfg.DATA.GRID_SIZE, cfg.DATA.GRID_SIZE, 4],
							dtype=cfg.MODEL.FLOATX)

	# TODO not good for low number of pool sizes, e.g. 8/3 -> 2,2,4 instead of 3,3,2
	sizes = cfg.TRAIN.POOL_SIZE // len(emoji_strings)
	for idx, emoji_str in enumerate(emoji_strings):
		cur_size = sizes

		# if last element, take rest of space
		if emoji_strings.index(emoji_str) == len(emoji_strings) - 1:
			cur_size = cfg.TRAIN.POOL_SIZE - (len(emoji_strings) - 1) * sizes

		(cur_x, cur_y), (_,_) = get_single_emoji_edge_data(emoji_str, amount=cur_size)

		full_batch_x[idx*sizes : idx*sizes + cur_size] = cur_x
		full_batch_y[idx*sizes : idx*sizes + cur_size] = cur_y

	return (full_batch_x, full_batch_y), (None, None)




# TODO not sure where to fit seeds, but here seems k
def init_seed():
	""" Get initial World seed with single alive cell in the middle """
	assert cfg.MODEL.CHANNEL_N >= 3, "CHANNEL_N has to be greater 3 to account for rgb"
	seed = np.zeros([1, cfg.DATA.GRID_SIZE, cfg.DATA.GRID_SIZE, cfg.MODEL.CHANNEL_N], cfg.MODEL.FLOATX)

	# Selecting the middle pixel and setting it to black, no just activating middle cells
	# TODO (why 3:)-> make cells alive and motivate computation
	seed[:, cfg.DATA.GRID_SIZE//2, cfg.DATA.GRID_SIZE//2, 3:] = 1.0

	return seed

def random_seed():
	seed = np.random.uniform(0, 1, [1, cfg.DATA.GRID_SIZE, cfg.DATA.GRID_SIZE, cfg.MODEL.CHANNEL_N])

	# Selecting the middle pixel and setting it to black
	# seed[:, cfg.TARGET_SIZE//2, cfg.TARGET_SIZE//2, :3] = 1.0

	return seed

emoji_dict = {

	# more complex things:
	"lizard": "1f98e",
	"t_rex": "1f996",
	"frog": "1f438",
	"whale": "1f40b",
	"tropical_fish": "1f420",
	"test_tube": "1f9ea",
	"alembic": "2697",
	"dna": "1f9ec",
	"microscope": "1f52c",
	"satelite_antenna": "1f4e1",
	"syringe": "1f489",
	"dagger": "1f5e1",
	"crossed_sword": "2694",
	"water_pistol": "1f52b",
	"shield": "1f6e1",
	"wrench": "1fa9a",
	"books": "1f4da",
	"ledger": "1f4d2",
	"film_projector": "1f4fd",
	"camera_with_flash": "1f4f8",
	"magnifying_glass": "1f50d",
	"trackball": "1f5b2",
	"electric_plug": "1f50c",
	"laptop": "1f4bb",
	"floppy_disk": "1f4be",
	"telephone": "260e",
	"mobile_phone": "1f4f1",
	"pager": "1f4df",
	"saxophone": "1f3b7",
	"trumpet": "1f3ba",
	"violine": "1f3bb",
	"drum": "1f941",
	"control_knobs": "1f39b",
	"radio": "1f4fb",
	"loudspeaker": "1f4e2",
	"bell_with_slash": "1f515",


	# Symbols
	"double_exclamation_mark": "203c",
	"heavy_check_mark": "2714",
	"heart": "2764",
	"hollow_red_circle": "2b55",
	"cross_mark": "274c",
	"curly_loop": "27b0",
	"plus": "2795",
	"multiply" :"2716",
	"minus": "2796",
	"divide": "2797",
	"exclamation_question_mark": "2049",
	"curly_loop": "27b0",
	"eight_pointed_star": "2733",
	"copyright": "00a9",
	"green_circle": "1f7e2"
}