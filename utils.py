"""
Utils file for all the nice functions to use
Might divide it into extra visualization file for all the Vids and imgs stuff
And maybe for the graphs
"""
# My Files
import config as cfg

import tensorflow as tf
 #from tensorflow.keras.layers import Conv2D
import numpy as np

import io
import time

import copy

# Video and visualization stuff
import tqdm
import PIL.Image, PIL.ImageDraw

from IPython.display import Image
import moviepy.editor as mvp

import matplotlib.pyplot as plt
from visdom import Visdom

# For loading and saving
import os
import json
import pickle 
from datetime import datetime

total_time_dict = {}


grid_plots_simple_nov = {
	"2020_11_12/15_32_39": [1,1], "2020_11_12/15_57_07": [2,1], "2020_11_12/16_22_01": [3,1],
	"2020_11_13/08_29_06": [4,1], "2020_11_13/08_59_37": [1,2], "2020_11_13/09_26_06": [2,2],
	"2020_11_13/10_02_12": [3,2], "2020_11_13/10_37_47": [4,2], "2020_11_14/12_06_40": [1,3],
	"2020_11_14/13_09_57": [2,3], "2020_11_14/13_59_28": [3,3], "2020_11_14/14_42_17": [4,3],
	"2020_11_14/12_34_33": [1,4], "2020_11_12/16_49_50": [2,4], "2020_11_14/15_20_18": [3,4],
	"2020_11_13/14_01_34": [4,4]
}


# TODO this function could be in an superclass for both trainers...
def save_model_and_plot(trainer, path, name=""):

	# Save current model
	trainer.save_weights(path=path, name=name + "_model")

	# Save full loss plot
	plot = trainer.plot_losses()

	save_fig(plot, name=name + "_fig", as_img=True, path=path)
	save_cfg(path=path, name=name + "_")

	
def background_options():
	""" tf and np configs to be run in the background"""
	tf.keras.backend.set_floatx(cfg.MODEL.FLOATX)

	if cfg.EXTRA.LOG_LAYERS:
		tf.config.run_functions_eagerly(True)
	else:
		tf.config.run_functions_eagerly(False)

	if cfg.EXTRA.FIXED_SEED:
		tf.random.set_seed(cfg.EXTRA.FIXED_SEED)
		np.random.seed(cfg.EXTRA.FIXED_SEED)

def get_current_time():
	return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def get_summary_writers():
	path = get_full_log_path() + '/logs/'
	
	train_log_dir = path + '/train'
	val_log_dir = path + '/val'
	test_log_dir = path + '/test'

	train_summary_writer = tf.summary.create_file_writer(train_log_dir)
	val_summary_writer = tf.summary.create_file_writer(val_log_dir)
	test_summary_writer = tf.summary.create_file_writer(test_log_dir)

	return train_summary_writer, val_summary_writer, test_summary_writer

def get_auto_title():
	""" generates automatic title based on config """
	pass

def get_cfg_infos():
	""" creates small ish title str with important informations """
	# Hotfix to shorten the DATA dict string with too many targets
	data_dict = copy.deepcopy(cfg.DATA.__dict__)
	data_dict["TARGETS"] = "many"

	# Add session id to str if not in extra dict
	if "SESSION_ID" not in list(cfg.EXTRA.__dict__.keys()):
		cfg.EXTRA.__dict__["SESSION_ID"] = cfg.EXTRA.SESSION_ID

	full_str = str(cfg.WORLD.__dict__) + str(data_dict) + str(cfg.MODEL.__dict__) + \
						 str(cfg.TRAIN.__dict__) + str(cfg.EXTRA.__dict__)

	# arbitrary len of str, then too big
	if "test" not in cfg.settings_name and len(full_str) > 100:
		return cfg.settings_name + ": " + cfg.EXTRA.SESSION_ID

	return full_str

def shape(arr):
	""" Returns shape of array, len if list"""
	if type(arr) == list:
		return [len(arr)]
	else:
		return arr.shape

def save_img_to_file(array_img, path):
	""" saves image to file """
	np_img = np.uint8(array_img * 255.)
	img = PIL.Image.fromarray(np_img) 
	img.save(path)


def mmm(arr):
	return np.amin(arr), np.mean(arr), np.amax(arr)

# Wrapper to time functions
# TODO save times of methods to some outside arary to look at later
def timeit(method):
	if not cfg.EXTRA.USE_TIMER:
		def wrapper(*args, **kwargs):
			return method(*args, **kwargs)
	else:
		def wrapper(*args, **kwargs):
			start = time.time()
			result = method(*args, **kwargs)
			total = time.time() - start


			# Add to list of times 
			if not method.__name__ in total_time_dict.keys():
				total_time_dict[method.__name__] = []
			total_time_dict[method.__name__].append(total)

			# print(f"Execution time of {method.__name__} ", total)
			return result
	return wrapper


# Visdom stuff
class VisdomLinePlotter(object):
		"""Plots to Visdom"""
		def __init__(self, env_name='main'):
			self.viz = Visdom()
			self.env = env_name
			self.plots = {}

		def plot(self, var_name, split_name, title_name, x, y):
			if var_name not in self.plots:
				self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
				legend=[split_name],
								title=title_name,
								xlabel='Epochs',
								ylabel=var_name
						))
			else:
				self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


def color_labels(x, y_pic, disable_black=False, dtype=tf.uint8):
	# works for shapes of x [b, r, c] and [r, c]
	black_and_white = tf.fill(list(x.shape) + [2], 0.01)
	is_gray = tf.cast(x > 0.1, tf.float32)
	is_not_gray = 1. - is_gray

	y_pic = y_pic * tf.expand_dims(is_gray, -1) # forcibly cancels everything outside of it.
	
	# if disable_black, make is_gray super low.
	if disable_black:
		is_gray *= -1e5
		# this ensures that you don't draw white in the digits.
		is_not_gray += is_gray

	# TODO Background white thingy...
	bnw_order = [is_gray, is_not_gray] if cfg.BACKGROUND_WHITE else [is_not_gray, is_gray]
	black_and_white *= tf.stack(bnw_order, -1)

	rgb = tf.gather(
			cfg.COLOR_LOOKUP,
			tf.argmax(tf.concat([y_pic, black_and_white], -1), -1))
	if dtype == tf.uint8:
		return tf.cast(rgb, tf.uint8)
	else:
		return tf.cast(rgb, dtype) / 255.


def ensure_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)


# Todo what is that? -> I think just making the picture bigger
def zoom(img, scale=4, axis_offset=0):
	img = np.repeat(img, scale, 0 + axis_offset)
	img = np.repeat(img, scale, 1 + axis_offset)
	return img

def np2pil(a):
	if a.dtype in [np.float32, np.float64]:
		a = np.uint8(np.clip(a, 0, 1)*255)
	return PIL.Image.fromarray(a)

def imwrite(f, a, fmt=None):
	a = np.asarray(a)
	if isinstance(f, str):
		fmt = f.rsplit('.', 1)[-1].lower()
		if fmt == 'jpg':
			fmt = 'jpeg'
		f = open(f, 'wb')
	np2pil(a).save(f, fmt, quality=95)

def imencode(a, fmt='jpeg'):
	a = np.asarray(a)
	if len(a.shape) == 3 and a.shape[-1] == 4:
		fmt = 'png'
	f = io.BytesIO()
	imwrite(f, a, fmt)
	return f.getvalue()

def im2url(a, fmt='jpeg'):
	encoded = imencode(a, fmt)
	base64_byte_string = base64.b64encode(encoded).decode('ascii')
	return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

# TODO what does this?
def tile2d(a, w=None):
	a = np.asarray(a)
	if w is None:
		w = int(np.ceil(np.sqrt(len(a))))
	th, tw = a.shape[1:3]
	pad = (w-len(a))%w
	a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
	h = len(a)//w
	a = a.reshape([h, w]+list(a.shape[1:]))
	a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
	return a


def classify_and_show(ca, x, disable_black=False):
	# TODO might be not usefull
	return zoom(tile2d(classify_and_color(ca, x, disable_black)), scale=8)
		

def classify_and_color(ca, x, disable_black=False):
	return color_labels(
			x[:,:,:,0], ca.classify(x), disable_black, dtype=tf.float32)

def generate_tiled_figures(figures, fade_by=0.1):
	tiled_pool = tile2d(figures)
	fade_sz = int(tiled_pool.shape[0] * fade_by)
	fade = np.linspace(1.0, 0.0, fade_sz)
	ones = np.ones(fade_sz) 
	tiled_pool[:, :fade_sz] += (-tiled_pool[:, :fade_sz] + ones[None, :, None]) * fade[None, :, None] 
	tiled_pool[:, -fade_sz:] += (-tiled_pool[:, -fade_sz:] + ones[None, :, None]) * fade[None, ::-1, None]
	tiled_pool[:fade_sz, :] += (-tiled_pool[:fade_sz, :] + ones[:, None, None]) * fade[:, None, None]
	tiled_pool[-fade_sz:, :] += (-tiled_pool[-fade_sz:, :] + ones[:, None, None]) * fade[::-1, None, None]
	return tiled_pool

def generate_pool_figures(ca, pool, step_i):
	tiled_pool = tile2d(classify_and_color(ca, pool.x))
	fade = np.linspace(1.0, 0.0, 72)
	ones = np.ones(72) 
	tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[None, :, None] 
	tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[None, ::-1, None]
	tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[:, None, None]
	tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[::-1, None, None]
	# imwrite('train_log/%04d_pool.jpg'%step_i, tiled_pool)

def make_run_videos(ca, num_steps, eval_bs, prefix, disable_black=False):
	new_idx = np.random.randint(0, x_train.shape[0]-1, size=eval_bs)
	x = ca.initialize(x_train[new_idx])
	frames = []
	with VideoWriter(prefix + ".mp4") as vid:
		# Not needed
		# slider = PIL.Image.open("mnist_slider.png")
		for i in tqdm.trange(-1, num_steps):
			if i == -1:
				image = zoom(tile2d(classify_and_color(ca, x, disable_black=False)), scale=2)
			else:
				if i == num_steps//2:
					# then mutate
					new_idx = np.random.randint(0, x_train.shape[0]-1, size=eval_bs)
					new_x = x_train[new_idx]
					new_x = tf.reshape(new_x, [eval_bs, 28, 28, 1])
					mutate_mask = tf.cast(new_x > 0.1, tf.float32)
					x = tf.concat([new_x, x[:,:,:,1:] * mutate_mask], -1)
				x = ca(x)
				image = zoom(tile2d(classify_and_color(ca, x, disable_black=disable_black)), scale=2)
			vis_extended = np.concatenate((image, np.ones((86, image.shape[1], 3))), axis=0) 
			im = np.uint8(vis_extended*255)
			im = PIL.Image.fromarray(im)
			# not needed
			# im.paste(slider, box=(0, image.shape[0]+20))
			draw = PIL.ImageDraw.Draw(im)
			p_x = 3+(((image.shape[1]-5-3)/num_steps)*i)
			draw.rectangle([p_x, image.shape[0]+21, p_x+5, image.shape[0]+42], fill="#434343bd")
			vid.add(np.uint8(im))


def load_full_save(date, time):
	""" Returns class informations from pickle file from cfg.EXTRA.LOG_PATH + data + time """
	full_path = cfg.EXTRA.LOG_PATH + date + "/" + time + "/"
	if not os.path.isdir(full_path):
		raise ValueError(f"Path : {full_path} does not exists")

	all_class_instances = []
	with open(full_path + "full_config.pkl", "rb") as f:
		# Iterates through all cfg classes specified in cfg file
		for config_name in cfg.ALL_CONFIG_CLASSES:
			x = pickle.load(f)
			# If loading old versions, this has to be removed manually!
			assert type(x) == type(getattr(cfg, config_name)), f"{type(x)} of {x} not correct"
			all_class_instances.append(x)
	return all_class_instances


# Todo not happy about placement, in utils would create double import?
def get_session_id():
	now = datetime.now()
	now_date = f"{now.year}_{now.month:02d}_{now.day:02d}"
	now_time = f"{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
	return now_date + "/" + now_time


def read_json(filename):
	with open(filename) as f:
		data = json.load(f)
	return data


def get_full_log_path():
	""" returns str for path to log directory of current session id"""
	# First path for date
	cur_path = cfg.EXTRA.LOG_PATH + cfg.EXTRA.SESSION_ID.split("/")[0] + "/"
	if not os.path.isdir(cur_path):
		os.mkdir(cur_path)

	# Ad second path for time
	cur_path += cfg.EXTRA.SESSION_ID.split("/")[1] + "/"
	if not os.path.isdir(cur_path):
		os.mkdir(cur_path)

	return cur_path

def save_cfg(path=None, name=""):
	""" 
	Saves all config objects in bytes and all changed information as jsons
	At LOG_PATH + date + time
	"""

	if path is None:
		path = get_full_log_path()

	# Open and write into 'json' (not 100% json)
	with open(path + name + "diff.json", "w") as f:
		for config_name in cfg.ALL_CONFIG_CLASSES:
			f.write(f'{{"{config_name}":')
			json.dump(getattr(cfg, config_name).__dict__, f)
			f.write("}\n")

	# Open and write into pickle file
	with open(path + name + "full_config.pkl", "wb") as f:
		for config_name in cfg.ALL_CONFIG_CLASSES:
			pickle.dump(getattr(cfg, config_name), f)

	print("Successfully saved configs at: ", path)


# Todo this could be done as given a list of types like ["json", "svg"]
def save_fig(fig, name="fig", as_json=True, as_img=False, as_pdf=False, path=None):

	if path is None:
		path = cfg.EXTRA.LOG_PATH + cfg.EXTRA.SESSION_ID + "/"

	if as_json:
		fig.write_json(path + name + ".json")

	if as_img:
		full_path = path + name + ".png"
		try:
			fig.write_image(full_path)
		except ValueError:
			print("VALUE ERROR while writing the png")

	if as_pdf:
		full_path = path + name + ".pdf"
		try:
			fig.write_image(full_path)
		except ValueError:
			print("VALUE ERROR while writing the png")



	print("Successfully saved figure at: ", full_path)


def get_all_figs(sess_id, name="fig_"):
	path = cfg.EXTRA.LOG_PATH + "/" + sess_id + "/"

	all_files = os.listdir(path)
	file_names = [file for file in all_files if name in file and "json" in file]

	figs = []
	for file_name in file_names:
		figs.append(read_json(path + file_name))

	return figs

def get_all_figs_by_path(path, name="fig"):
	all_files = os.listdir(path)
	file_names = [file for file in all_files if name in file and "json" in file]

	figs = []
	for file_name in file_names:
		figs.append(read_json(path + file_name))
	return figs

def open_cfg_from_path(path, name="full_config.pkl"):
  """ Open cfg.pkl file from path returns 4 dicts"""
  out_arr = []
  with open(path + name, "rb") as f:
    while 1:
      try:
        data = pickle.load(f)
        
        out_arr.append(object_to_dict(data))
      except (EOFError, pickle.UnpicklingError):
        break
        
  return out_arr
  
def object_to_dict(obj):
  """ Turns object into dictionary"""
  res = {}
  for attr in dir(obj):
    if "__" in attr:
      continue
    else:
      res[attr] = getattr(obj, attr)
  return res