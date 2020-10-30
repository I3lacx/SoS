"""
Global Configurations
"""
# TODO that does not fit here, but Color lookup... Maybe change that
import tensorflow as tf


# ----- Global Variables -------
CHANNEL_N = 16 # Number of CA state channels (includes rgb for growing)
BATCH_SIZE = 8
POOL_SIZE = BATCH_SIZE * 10
CELL_FIRE_RATE = 1  # Random chance if single cell will update or not, as there is no global clock in a cell system

HIDDEN_LAYERS = 0 	# Number of hidden layers {0,1,...}
HIDDEN_FILTER_SIZE = 128 # Number of filters in hidden layer

MODEL_TASK = 'growing' #@param ["growing", "classify"]
MODEL_TYPE = '3'  #@param ['1 naive', '2 persistent', '3 mutating']
LOSS_TYPE = "l2"  #@param ['l2', 'ce']
ADD_NOISE = "True"  #@param ['True', 'False']
CA_STEP_RANGE = [50, 60]	# RANGE of CA steps until update function is applied

# Only for growing Model, which emoji to grow, idk how to get those Ascis
TARGET_EMOJI = "lizard" # in string form, defined in datasets.Emoji
TARGET_SIZE = 40

# ----- Learning ------
LR = 1e-3
LAYER_NORM = True


USE_TIMER = True		# Prints and (saves) times of methods

# Environment Size is dimensions that are not influenced by updates
# E.g. 1 for gray_scale, 3 for RGB, 0 for growing emojis
ENV_SIZE = 0

SAVE_MODEL = False

USE_PATTERN_POOL, MUTATE_POOL = {
		'1': (False, False),
		'2': (True, False),
		'3': (True, True)
		}[MODEL_TYPE]
ADD_NOISE = ADD_NOISE == 'True'

# FOR TRAINING
NUM_TRAINING_STEPS = 1000

# Not sure what this is, but needed in utils, could be related to MNIST
BACKGROUND_WHITE = True
COLOR_LOOKUP = tf.constant([
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
						[0, 0, 0], # This is the default for digits.
						[255, 255, 255] # This is the background.
						])
