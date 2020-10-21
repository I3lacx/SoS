"""
Containing all stuff about each dataset
Should later be modified into classes, so that I load 
each Dataset as one class with their respective function
"""
import numpy as np
import tensorflow as tf
import time

import utils

# Only for samples_str
import json 
# --- MNIST --- (later as Class maybe)



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
		idx = np.random.choice(self._size, n, False)
		batch = {k: getattr(self, k)[idx] for k in self._slot_names}
		batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
		return batch

	def commit(self):
		for k in self._slot_names:
			getattr(self._parent, k)[self._parent_idx] = getattr(self, k)
