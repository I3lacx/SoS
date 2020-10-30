"""
Display module in control of all the things regarding showing imgs, showing plots etc.
Some things might still be in utils, will be moved over here over time.
"""
import numpy as np

import utils 
import config as cfg

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from IPython.display import Image, HTML, clear_output, Video, display

import matplotlib.pyplot as plt


class VideoWriter:
	# name is just reused, does not actually autoplay...
	def __init__(self, filename="_autoplay.mp4", fps=30.0, **kw):
		self.writer = None
		self.params = dict(filename=filename, fps=fps, **kw)

	def add(self, img):
		img = np.asarray(img)
		if self.writer is None:
			h, w = img.shape[:2]
			self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
		if img.dtype in [np.float32, np.float64]:
			img = np.uint8(img.clip(0, 1)*255)
		if len(img.shape) == 2:
			img = np.repeat(img[..., None], 3, -1)
		if len(img.shape) == 3 and img.shape[-1] == 4:
			img = img[..., :3] * img[..., 3, None]
		self.writer.write_frame(img)

	def close(self):
		if self.writer:
			self.writer.close()

	def __enter__(self):
		return self

	def __exit__(self, *kw):
		self.close()


def show_simple_run(x0, ca, num_steps=20, batch=0):
	x = x0
	with VideoWriter("_autoplay.mp4") as vid:
		for i in range(num_steps):
			x = ca(x)
			vid.add(utils.zoom(ca.classify(x)[batch]))
	run = Video("./_autoplay.mp4", width=320, height=220, html_attributes="autoplay controls", embed=True)
	return run

def clear():
	""" Calls IPython clear_output function to clear current outputs """
	clear_output(wait=True)

def show(obj):
	display(obj)

def plot_loss(loss_log, log_scale=True, return_plot=False):
	plt.figure(figsize=(10, 4))
	plt.title('Loss history (log10)')
	if log_scale:
		plt.yscale('log')
	plt.plot(np.log10(loss_log), '.', alpha=0.5)

	if return_plot:
		return plt.gcf()
	plt.show()
	# This does not remove old plots and they all pop up in the end..
	# Could be usefull when used intentional
	# display(plt.gcf())


def imshow(a, fmt='jpeg'):
	# TODO likely need to include this again if using Notebook
	# display is a Jupyter widget
	# plt.imshow(utils.imencode(a))
	display(Image(data=utils.imencode(a, fmt), embed=True))
	# plt.imshow(a)
	# plt.show()


def visualize_batch(ca, x0, x, step_i):
	if cfg.MODEL_TASK == "classify":
		vis0 = np.hstack(classify_and_color(ca, x0).numpy())
		vis1 = np.hstack(classify_and_color(ca, x).numpy())
		vis = np.vstack([vis0, vis1])
		# imwrite('train_log/batches_%04d.jpg'%step_i, vis)
		print('batch (before/after):')
	elif cfg.MODEL_TASK == "growing":
		# Classify in growing, returns RGB layers
		vis = np.hstack(ca.classify(x))
	else:
		raise ValueError()

	imshow(vis)

def vis_scatter(vis, x, title):
	""" Creates a simple scatter plot in visdom """
	if len(x.shape) == 1:
		x = x[:, None]

	assert len(x.shape) == 2 and x.shape[1] == 1, f"x shape: {x.shape} unsuported"

	# Create y values
	y = np.arange(0, x.shape[0])[:, None]
	z = np.hstack((y,x))

	opts = {'markersize':7, 'ylabel':'logloss', 'xlabel':'step',
		 'title':title, 'ytype':'log'}
	vis.scatter(z, opts=opts)

