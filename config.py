"""
Global Configurations
"""
# TODO that does not fit here, but Color lookup... Maybe change that
import tensorflow as tf


# ----- Global Variables -------
CHANNEL_N = 19 # Number of CA state channels
BATCH_SIZE = 16
POOL_SIZE = BATCH_SIZE * 10
CELL_FIRE_RATE = 0.5  # Random chance if single cell will update or not, as there is no global clock in a cell system

MODEL_TYPE = '3 mutating'  #@param ['1 naive', '2 persistent', '3 mutating']
LOSS_TYPE = "l2"  #@param ['l2', 'ce']
ADD_NOISE = "True"  #@param ['True', 'False']

# Environment Size is dimensions that are not influenced by updates
# E.g. 1 for gray_scale, 3 for RGB
ENV_SIZE = 1 


USE_PATTERN_POOL, MUTATE_POOL = {
		'1 naive': (False, False),
		'2 persistent': (True, False),
		'3 mutating': (True, True)
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
