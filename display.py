"""
Display module in control of all the things regarding showing imgs, showing plots etc.
Some things might still be in utils, will be moved over here over time.
"""
import numpy as np

import utils 
import config as cfg

import math
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from IPython.display import Image, HTML, clear_output, Video, display

# TODO plt should/could be fully removed
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly


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


def show_simple_run(x0, ca, num_steps=20, batch=0, only_alive_cells=False, white_background=False):
	x = x0
	with VideoWriter("_autoplay.mp4") as vid:
		for i in range(num_steps):
			x = ca(x)
			if only_alive_cells:
				res = utils.zoom((x[batch, :, :, 3].numpy() > 0.1).astype(np.float32))
				vid.add(res)
			else:
				vid.add(utils.zoom(rgba_to_rgb(ca.classify(x)[batch], white_background=white_background)))
	run = Video("./_autoplay.mp4", width=320, height=220, html_attributes="autoplay controls", embed=True)
	return run

def clear():
	""" Calls IPython clear_output function to clear current outputs """
	clear_output(wait=True)

def show(obj):
	display(obj)

def plot_loss(loss_log, log_scale=False, return_plot=False, plot_mean=True, name="", smoothed=0, title=""):
	""" Plot loss with plotly """
	# TODO plot only the mean if there are too many points -> performance

	if smoothed > 0:
		# kernel = [smoothed, 1-smoothed]
		# smoothed_y = np.convolve(kernel, loss_log, 'valid')
		# smoothed_y = np.insert(smoothed_y, 0, loss_log[0])
		smoothed_y = smooth(loss_log, weight=smoothed)
		fig = go.Scatter(y=smoothed_y, name=str(name+"_s"))
	else:
		fig = go.Scatter(y=loss_log, name=str(name))

	if return_plot:
		return fig

	plot = go.Figure(fig)

	if name:
		plot.update_layout(title=title)
	if log_scale:
		plot.update_yaxes(type="log")

	plot.show()

	# TODO i think this can be removed: Old conv filter
	"""
	# Create mean plot with line
	# basically a smoothing value higher -> more smooth
	step_size = 5
	kernel = 1/step_size * np.ones((step_size),dtype=np.float32)
	y_mean = np.convolve(kernel, loss_log, 'same')[step_size//2::step_size]
	x_vals = np.arange(1,len(loss_log),step_size)


	# Plot initial data points:
	fig = px.scatter(name=name, y=loss_log, opacity=0.5)
	fig.layout.title = {"text": "Loss log", "x":0.5, "xanchor":"center"}
	# Keep in mind:
	# {"text": "Loss log?", "font":{"family":"Open Sans", "size":24}, "x":0.5, "xanchor":"center"}

	if plot_mean:
		# Add second plot
		fig2_data = px.scatter(y=y_mean, x=x_vals).data[0]
		fig2_data.mode = "lines"
		fig2_data.marker.color="red"
		fig.add_trace(fig2_data)

	if return_plot:
		return fig

	# Show figure
	fig.show()
	# TODO log_scale
	"""


# TODO does not seem suuper efficient -> also should be in utils not in display
def smooth(scalars, weight):  # Weight between 0 and 1
	last = scalars[0]  # First value in the plot (first timestep)
	smoothed = list()
	for point in scalars:
		if math.isnan(last):
			last = 0
			smoothed_val = point
		smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
		smoothed.append(smoothed_val)                        # Save it
		last = smoothed_val                                  # Anchor the last smoothed value

	return smoothed


def plot_session(sess_id):
	""" plots several lossos of a single session with different runs into a single plot """
	figs = []

	fig = utils.read_json()

import textwrap

def plot_losses(loss_logs, log_scale=False, return_plot=False, plot_mean=True,
 smoothing_weight=0.8, title_text=utils.get_cfg_infos(), textwrap_width=120):
	""" loss logs to plot in a single graph. 
	Expected shape: [n, number_losses] """
	
	loss_logs = np.array(loss_logs)

	fig = go.Figure()
	names = ["total", "seed", "pool"]
	for i in range(np.shape(loss_logs)[1]):
		trace = plot_loss(loss_logs[:,i], return_plot=True, plot_mean=False, name=names[i])
		smooth = plot_loss(loss_logs[:,i], return_plot=True, name=names[i], smoothed=smoothing_weight)
		fig.add_trace(trace)
		fig.add_trace(smooth)

	fig.layout.title.text = "<br>".join(textwrap.wrap(title_text, width=textwrap_width))
	fig.layout.title.font.size = 12
	# fig.layout.uniformtext.mode = "show"

	if log_scale:
		fig.update_yaxes(type="log")
	"""
	TODO insert layout into there?
	fig_dict = dict({
		"data": data,
		"layout":{"title":{"text":"Several Losses", "x":0.5, "xanchor":"center"}}
		})

	fig = go.Figure(fig_dict)
	"""
	# fig = px.scatter(data)
	# fig.layout.title = {"text": "Several Losses", "x":0.5, "xanchor":"center"}

	"""
	plt.plot(loss_logs[:,0])
	plt.show()
	plt.plot(loss_logs[:,1])
	plt.show()
	plt.plot(loss_logs[:,2])
	plt.show()
	"""

	if return_plot:
		return fig

	# Show figure
	fig.show()
	# TODO log_scale


def imshow(a, fmt='jpeg'):
	# TODO likely need to include this again if using Notebook
	# display is a Jupyter widget
	# plt.imshow(utils.imencode(a))
	display(Image(data=utils.imencode(a, fmt), embed=True))
	# plt.imshow(a)
	# plt.show()


def rgba_to_rgb(rgba_img, white_background=False):
	''' Inverts alpha channel to add to rgb on white '''
	rgb = np.clip(rgba_img[...,:3], 0.0, 1.0)
	a = np.clip(rgba_img[...,3:4], 0.0, 1.0)

	if white_background:
		return 1.0 - a + rgb
	else:
		return rgba_img[...,:3] * rgba_img[...,3:4]


def visualize_batch(ca, x0, x, step_i):
	if cfg.WORLD.TASK["TASK"] == "classify":
		vis0 = np.hstack(classify_and_color(ca, x0).numpy())
		vis1 = np.hstack(classify_and_color(ca, x).numpy())
		vis = np.vstack([vis0, vis1])
		# imwrite('train_log/batches_%04d.jpg'%step_i, vis)
		print('batch (before/after):')
	elif cfg.WORLD.TASK["TASK"] == "growing":
		# Classify in growing, returns RGBA layers
		vis_x = np.hstack(ca.classify(x))
		vis_x0 = np.hstack(ca.classify(x0))
		print(vis_x.shape)
		vis = np.vstack([vis_x0, vis_x])
	else:
		raise ValueError()

	imshow(vis)

def visdom_plotly(vis, fig):
	""" draws visdom plotly figure """
	vis.plotlyplot(fig)


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


def avg_plot(figs, idx, name, return_plot=True, showlegend=True, color=None):
	""" calculates mean and std from a single index from figs """
	figs_mean = np.mean([fig["data"][idx]["y"] for fig in figs], axis=0)
	figs_std = np.std([fig["data"][idx]["y"] for fig in figs], axis=0)

	figs_x_single = np.arange(figs_mean.shape[0])
	figs_x = np.concatenate((figs_x_single, figs_x_single[::-1]))
	figs_y_upper = figs_mean + figs_std
	figs_y_lower = figs_mean - figs_std
	figs_y = np.concatenate((figs_y_upper, figs_y_lower[::-1]))

	std_marker_color = "rgba(100, 30, 150, 0.5)"
	# Using legendgroup will merge plots on a huge subplotplot level
	mean_plot = go.Scatter(y=figs_mean, name=name+"_mean", legendgroup=name+"_mean", showlegend=showlegend, marker_color=color)
	std_plot = go.Scatter(x=figs_x, y=figs_y, name=name+"_std", fill='toself', legendgroup=name+"_std", showlegend=showlegend, marker_color=color, fillcolor=color, opacity=0.5)

	# std_plot.update_traces(marker_opacity=0.5)

	if return_plot:
		return mean_plot, std_plot
	else:
		go.Figure([mean_plot, std_plot]).show()


def plot_avg_session(sess_id, title="", showlegend=True):
	""" gets all avg plots from data_idx (realated to lossos)"""
	figs = utils.get_all_figs(sess_id)

	data_idx = [1, 3, 5]
	names = ["total", "seed", "pool"]

	if title == "":
		# TODO this is disgusting code
		with open(cfg.EXTRA.LOG_PATH + sess_id + "/diff.json") as f:
			file_arr = f.read().split("\n")

		for i in range(len(file_arr)):
  			title += file_arr[i][9:-1]

	color_arr = plotly.colors.DEFAULT_PLOTLY_COLORS
	# color_arr = plotly.colors.sequential.Plasma[::3]

	fig = go.Figure()
	for idx,val in enumerate(data_idx):
		mean_plot, std_plot = avg_plot(figs, val, names[idx], showlegend=showlegend, color=color_arr[idx])
		fig.add_trace(mean_plot)
		fig.add_trace(std_plot)
		fig.layout.title.text = "<br>".join(textwrap.wrap(title, width=120))
		fig.layout.title.font.size = 12
	return fig

def plot_collection_of_avg_plots(plots, render_in_browser=True):
	""" Plots a collection of avergae plots from dict of plot ids from plots
		This function is not generalized! """

	# This will default all coming rendering of plotly to the browser!
	if render_in_browser:
		pio.renderers.default = "browser"

	fig = make_subplots(rows=4, cols=4, x_title="HIDDEN_FILTER_SIZE [16,128]", y_title="CHANNEL_N [8,64]")

	for idx, key in enumerate(plots.keys()):
		if idx == 0:
			plot = plot_avg_session(key, showlegend=True).data
		else:
			plot = plot_avg_session(key, showlegend=False).data

		fig.add_traces(plot, rows=plots[key][0], cols=plots[key][1])

	fig.update_yaxes(range=[0,300])
	fig.update_layout(height=900, width=1300, title_text="Multiple Subplots with Titles")

	fig.show()