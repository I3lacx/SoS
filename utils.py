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


def get_auto_title():
	""" generates automatic title based on config """
	pass

def get_cfg_infos():
	""" creates small ish title str with important informations """
	full_str = str(cfg.WORLD.__dict__) + str(cfg.MODEL.__dict__) + \
						 str(cfg.TRAIN.__dict__) + str(cfg.EXTRA.__dict__)
	return full_str
	# return full_str

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
def zoom(img, scale=4):
	img = np.repeat(img, scale, 0)
	img = np.repeat(img, scale, 1)
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

def save_cfg():
	""" 
	Saves all config objects in bytes and all changed information as jsons
	At LOG_PATH + date + time
	"""

	cur_path = get_full_log_path()
	
	# Open and write into 'json' (not 100% json)
	with open(cur_path + "diff.json", "w") as f:
		for config_name in cfg.ALL_CONFIG_CLASSES:
			f.write(f'{{"{config_name}":')
			json.dump(getattr(cfg, config_name).__dict__, f)
			f.write("}\n")

	# Open and write into pickle file
	with open(cur_path + "full_config.pkl", "wb") as f:
		for config_name in cfg.ALL_CONFIG_CLASSES:
			pickle.dump(getattr(cfg, config_name), f)

	print("Successfully saved configs at: ", cur_path)


def save_fig(fig, name="fig", as_json=True, as_img=False):
	full_path = cfg.EXTRA.LOG_PATH + cfg.EXTRA.SESSION_ID + "/"

	if as_json:
		fig.write_json(full_path + name + ".json")

	if as_img:
		try:
			fig.write_image(full_path + name + ".png")
		except ValueError:
			print("VALUE ERROR while writing the png")


def get_all_figs(sess_id, name="fig_"):
  path = cfg.EXTRA.LOG_PATH + "/" + sess_id + "/"
  
  all_files = os.listdir(path)
  file_names = [file for file in all_files if name in file and "json" in file]
  
  figs = []
  for file_name in file_names:
    figs.append(read_json(path + file_name))
    
  return figs



def create_js_drawing():
	# TODO this does not work, just put in utils and remove from main screen
	raise NotImplementedError("Not working, as js errors, here to be somewhere")
	# Careful when running this as js can destroy the whole jupyter Notebook!

	#@title TensorFlow.js Demo Full whiteboard {run:"auto", vertical-output: true}
	#@markdown Select "CHECKPOINT" model to load the checkpoint created by running 
	#@markdown cells from the "Training" section of this notebook.
	#@markdown Technical note: CE models should be rendered differently to avoid
	#@markdown black pixels showing for low magnitude.
	import IPython.display
	import glob
	import tensorflow as tf

	# TODO this loading could be in utils:
	samples_str = datasets.get_samples_str(x_train, y_train)

	#@markdown draw with left click, hold shift for erasing

	# Load model from ckpt:
	# glob and stuff to load last json file (highest num_steps, only works with single model name here)
	last_checkpoint_fn = sorted(glob.glob('models/*.json'))[-1]
	model_str = open(last_checkpoint_fn).read()


	data_js = '''
		window.GRAPH_URL = URL.createObjectURL(new Blob([`%s`], {type: 'application/json'}));
		window.SAMPLES = %s
	'''%(model_str, samples_str)

	display(IPython.display.Javascript(data_js))

	IPython.display.HTML(js_script_drawing)



js_script_drawing = '''
<script src=\"https://unpkg.com/@tensorflow/tfjs@latest/dist/tf.min.js\"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cash/4.1.2/cash.min.js"></script>
<div class="slidecontainer">
		brushSize:
		<input type="range" min="1" max="10" value="4" class="slider" id="brushSlider">
		<span id='radius'>2.5</span>
</div>
<canvas id='canvas' style="border: 1px solid black; image-rendering: pixelated;"></canvas>
<script>
import '@tensorflow/tfjs-backend-wasm';
	"use strict";

	// Adds the WASM backend to the global backend registry.

	// Set the backend to WASM and wait for the module to be ready.
const main = async () => {

	const sleep = (ms)=>new Promise(resolve => setTimeout(resolve, ms));
	
	const parseConsts = model_graph=>{
		const dtypes = {'DT_INT32':['int32', 'intVal', Int32Array],
										'DT_FLOAT':['float32', 'floatVal', Float32Array]};
		
		const consts = {};
		model_graph.modelTopology.node.filter(n=>n.op=='Const').forEach((node=>{
			const v = node.attr.value.tensor;
			const [dtype, field, arrayType] = dtypes[v.dtype];
			if (!v.tensorShape.dim) {
				consts[node.name] = [tf.scalar(v[field][0], dtype)];
			} else {
				const shape = v.tensorShape.dim.map(d=>parseInt(d.size));
				let arr;
				if (v.tensorContent) {
					const data = atob(v.tensorContent);
					const buf = new Uint8Array(data.length);
					for (var i=0; i<data.length; ++i) {
						buf[i] = data.charCodeAt(i);
					}
					arr = new arrayType(buf.buffer);
				} else {
					const size = shape.reduce((a, b)=>a*b);
					arr = new arrayType(size);
					arr.fill(v[field][0]);
				}
				consts[node.name] = [tf.tensor(arr, shape, dtype)];
			}
		}));
		return consts;
	}
	
	let paused = false;
	let visibleChannel = -1;
	let firingChance = 0.5;
	let drawRadius = 2.5;

	$('#brushSlider').on('input', e=>{
			drawRadius = parseFloat(e.target.value)/2.0;
			$('#radius').text(drawRadius);
	});

	const colorLookup = tf.tensor([
			[128, 0, 0],
			[230, 25, 75],
			[70, 240, 240],
			[210, 245, 60],
			[250, 190, 190],
			[170, 110, 40],
			[170, 255, 195],
			[165, 163, 159],
			[0, 128, 128],
			[128, 128, 0],
			[0, 0, 0], // This is the default for digits.
			[255, 255, 255] // This is the background.
			])

	let backgroundWhite = true;


	const run = async () => {
			const r = await fetch(GRAPH_URL);
			const consts = parseConsts(await r.json());

			//const samples = tf.tensor(SAMPLES);
			//console.log(samples);

			const model = await tf.loadGraphModel(GRAPH_URL);

			const samples = tf.tensor(SAMPLES);
			//const samples = tf.zeros([2,5, 28, 28]);

			console.log("Loaded model")
			Object.assign(model.weights, consts);
			// samples.gather(tf.range(0, 4, 1, 'int32')
			const D = 28 * 5;
			const flatState = tf.concat([
				samples.reshape([2,5, 28, 28])
					.transpose([0,2,1,3]).reshape([1, 2 * 28, D]),
				tf.zeros([1, D-(28*2), D])],1);
			console.log(flatState)
			const state = tf.variable(
				tf.concat([flatState.expandDims(3),
										tf.zeros([1, D, D, 19])], 3));
			const [_, h, w, ch] = state.shape;

			const scale = 8;

			const canvas = document.getElementById('canvas');
			const ctx = canvas.getContext('2d');
			canvas.width = w * scale;
			canvas.height = h * scale;

			const drawing_canvas = new OffscreenCanvas(w, h);
			const draw_ctx = drawing_canvas.getContext('2d');

			// Useful for understanding background color.
			
			//let blackAndWhite = tf.zeros();//.fill(0.01);
			let arr = new Float32Array(h * w * 2);
			arr.fill(0.01);
			const blackAndWhiteFull = tf.tensor(arr, [1,h,w,2], tf.float32)

			const drawCanvas = (imgd, e) => {
					var matrix = [];
					for(let i=0; i<imgd.width; i++) {
							matrix[i] = [];
							for(let j=0; j<imgd.height; j++) {
									let intensity = imgd.data[(imgd.height*j*4 + i*4)];
									// For drawing, we want to add shades of grey. For erasing, we don't.
									if (!e.shiftKey) {
										intensity *= (imgd.data[(imgd.height*j*4 + i*4 + 3)] / 255);
									}
									matrix[i][j] = intensity;
							}
					}

					tf.tidy(() => {
							const stroke = tf.tensor(matrix).transpose().toFloat().div(255.).expandDims(0).expandDims(3);
							const stroke_pad = tf.concat([stroke, tf.zeros([1, h, w, ch-1])], 3);
							const mask = tf.tensor(1.).sub(stroke);
							if (e.shiftKey) {
									state.assign(state.mul(mask));
							} else {
									state.assign(state.mul(mask).add(stroke_pad));
							}
					});

					// Then clear the canvas.
					draw_ctx.clearRect(0, 0, draw_ctx.canvas.width, draw_ctx.canvas.height);
			}

			const line = (x0, y0, x1, y1, r, e) => {
					draw_ctx.beginPath();
					draw_ctx.moveTo(x0, y0);
					draw_ctx.lineTo(x1, y1);
					draw_ctx.strokeStyle = "#ff0000";
					// Erasing has a much larger radius.
					draw_ctx.lineWidth = (e.shiftKey ? 5. * r : r);
					draw_ctx.stroke();

					const imgd = draw_ctx.getImageData(0, 0, draw_ctx.canvas.width, draw_ctx.canvas.height);
					drawCanvas(imgd, e);
			}


			const circle = (x, y, r, e) => {
					draw_ctx.beginPath();

					const drawRadius = (e.shiftKey ? 5. * r : r) / 3.;

					draw_ctx.arc(x, y, drawRadius, 0, 2 * Math.PI, false);
					draw_ctx.fillStyle = "#ff0000";
					draw_ctx.fill();
					draw_ctx.lineWidth = 1;
					draw_ctx.strokeStyle = "#ff0000";
					draw_ctx.stroke();

					const imgd = draw_ctx.getImageData(0, 0, draw_ctx.canvas.width, draw_ctx.canvas.height);
					drawCanvas(imgd, e);
			}

			const draw_r = 2.0;


			const getClickPos = e=>{
					const x = Math.floor((e.pageX-e.target.offsetLeft) / scale);
					const y = Math.floor((e.pageY-e.target.offsetTop) / scale);
					return [x, y];
			}

			let lastX = 0;
			let lastY = 0;

			canvas.onmousedown = e => {
					const [x, y] = getClickPos(e);
					lastX = x;
					lastY = y;
					circle(x,y,drawRadius, e);
			}
			canvas.onmousemove = e => {
					const [x, y] = getClickPos(e);
					if (e.buttons == 1) {
							line(lastX,lastY, x,y,drawRadius, e);
					}
					lastX = x;
					lastY = y;
			}
			const render = async () => {
				if (!paused) {
					tf.tidy(() => {
							state.assign(model.execute(
								{ x: state,
									fire_rate: tf.tensor(firingChance),
									manual_noise: tf.randomNormal([1, h, w, ch-1], 0., 0.02)},
								['Identity']));
					});
				}
				const imageData = tf.tidy(() => {
						let rgbaBytes;
						let rgba;
						if (visibleChannel < 0) {
								const isGray = state.slice([0,0,0,0],[1, h, w, 1]).greater(0.1).toFloat();
								const isNotGray = tf.tensor(1.).sub(isGray);

								const bnwOrder = backgroundWhite ?  [isGray, isNotGray] : [isNotGray, isGray];
								let blackAndWhite = blackAndWhiteFull.mul(tf.concat(bnwOrder, 3));

								const grey = state.gather([0], 3).mul(255);
								const rgb = tf.gather(colorLookup,
																			tf.argMax(
																			tf.concat([
									state.slice([0,0,0,ch-10],[1,h,w,10]),
									blackAndWhite], 3), 3));

								rgba = tf.concat([rgb, grey], 3)
						} else {
								rgba = state.gather([visibleChannel, visibleChannel, visibleChannel], 3)
									.pad([[0, 0], [0, 0], [0, 0], [0, 1]], 1).mul(255);
						}
						rgbaBytes = new Uint8ClampedArray(rgba.dataSync());

						return new ImageData(rgbaBytes, w, h);
				});
				const image = await createImageBitmap(imageData);
				//ctx.clearRect(0, 0, canvas.width, canvas.height);
				ctx.fillStyle = backgroundWhite ? "#ffffff" : "#000000";
				ctx.fillRect(0, 0, canvas.width, canvas.height);
				ctx.imageSmoothingEnabled = false;
				ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

				requestAnimationFrame(render);
			}
			render();
	}

	run();
}
main();
	//tf.setBackend('wasm').then(() => main());

	
</script>
'''