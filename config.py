""" 
Config file to define all configs, to change configs, change the input
dictionary for the Classes.
"""
# TODO remove!
import tensorflow as tf

# TODO not super fun...
import utils

class _ConfigModule:
	""" Template Class, will be used for each class """

	def __init__(self, params):
		# Only changes default params if specified in params
		self.set_params(params)

	def set_params(self, params):
		""" Sets all parameters to the values specified in configs """

		for key in params.keys():
			# Not needed, but makes sure that all parameters are defined above!
			assert hasattr(self, key)
			setattr(self, key, params[key])


class _World(_ConfigModule):
	""" Parameters for the World configuration """

	TASK = {"TASK":"growing", "TARGET":"lizard", "SIZE":30}
	SIZE = 40 	# Size of the whole grid, inside is the task, e.g. emoji size
	CELL_FIRE_RATE = 1
	CA_STEP_RANGE = [50, 60]
	ENV_SIZE = 0
	LIVING_MASK = True

	def __init__(self, params):
		super().__init__(params)


class _Model(_ConfigModule):
	""" Parameters for the Model configuration """
	CHANNEL_N = 16
	HIDDEN_LAYERS = 0
	HIDDEN_FILTER_SIZE = 128
	SKIP_CONNECTIONS = False
	LAST_LAYER_INIT = "ZEROS"
	BATCH_NORM = False

	def __init__(self, params={}):
		super().__init__(params)
		self.post_process()

	def post_process(self):
		pass


class _Training(_ConfigModule):
	""" Parameters for the Training configuration """

	NUM_TRAINING_STEPS = 1000 	# TODO not used I think
	BATCH_SIZE = 8
	POOL_SIZE = BATCH_SIZE * 10,
	FIXED_SEED_RATIO = None		# Fixing the seed ratio to float: (0,1] 

	POOL_TANH = False # Force pool values between (-1,1)

	# TRAINING_TYPE: '3', # Before: Model type
	# TODO conversion not here?
	USE_PATTERN_POOL =  True
	MUTATE_POOL = True

	LOSS_TYPE = "l2"
	ADD_NOISE = True # TODO used? -> I think first for MNIST
	LR = 1e-3
	LAYER_NORM = True

	def __init__(self, params={}):
		super().__init__(params)
		self.post_process()

	def post_process(self):
		pass


class _Extra(_ConfigModule):
	""" Extra configuraiots, not fittign other modules """
	USE_TIMER = True
	LOG_PATH = "logs/maybe/"
	SESSION_ID = None

	def __init__(self, params={}):
		super().__init__(params)


# Calls and creates all config objects, to be called after importing the module
# Session Id is set here to make it easier to recognize plots as they all have the id now.
world_dict = {"TASK": {"TASK":"growing", "TARGET":"double_exclamation_mark", "SIZE":20}, "SIZE":50}
model_dict = {}
train_dict = {"POOL_TANH": True, "MUTATE_POOL": False, "FIXED_SEED_RATIO": 1/8}
extra_dict = {"SESSION_ID": utils.get_session_id()}

WORLD = _World(world_dict)
MODEL = _Model(model_dict)
TRAIN = _Training(train_dict)
EXTRA = _Extra(extra_dict)

# This is used to save and load the classes
ALL_CONFIG_CLASSES = ["WORLD", "MODEL", "TRAIN", "EXTRA"]

# TODO Include?
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