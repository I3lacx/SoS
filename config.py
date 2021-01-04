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
	# tf initializer: [ZEROS, GLOROT, RANDOM_NORMAL]
	LAST_LAYER_INIT = "ZEROS" 
	BATCH_NORM = False
	UPDATE_GATE = False

	# TODO last layer init and gru and bias maybe better options
	GRU = False
	GRU_BIAS = True 
	SET_WORLD_STATE = False 	# on True this will not apply state += ds, but rather state = ds to the world state

	def __init__(self, params={}):
		super().__init__(params)
		self.post_process()

	def post_process(self):
		pass


class _Training(_ConfigModule):
	""" Parameters for the Training configuration """

	NUM_TRAIN_STEPS = 2000 	# TODO this value will be overwritten in notbeook, not nice...
	BATCH_SIZE = 8
	POOL_SIZE = BATCH_SIZE * 10,
	FIXED_SEED_RATIO = 1/8		# Fixing the seed ratio to float: (0,1] 

	POOL_TANH = False # Force pool values between (-1,1)

	""" Ratio of how long to initialize a warm up until the actual ratios kick in
		0 for no warm up, otherwise value between [0,1] How much of total steps """
	WARM_UP = 0. # 0.05 is decent other value, but still shit

	USE_PATTERN_POOL =  True
	MUTATE_POOL = False # 12.11 changed from True to False

	USE_Y_POOL = False	# Added option at 16.11

	LOSS_TYPE = "l2"
	ADD_NOISE = True # TODO YOS THIS IS USED DURING EACH CALL
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
	PRINT_LAYER = False
	LIST_OF_WEIGHT_NAMES = "all" # List containing the string names of weights to plot in tensorboard, 
		# "all" for every weight
	TENSORBOARD = False
	TB_GRAPH = False
	LOG_LAYERS = False # Logs output of layers (ugly), only works in eager mode

	def __init__(self, params={}):
		super().__init__(params)


# Calls and creates all config objects, to be called after importing the module
# Session Id is set here to make it easier to recognize plots as they all have the id now.
world_dict = dict()
model_dict = dict()
train_dict = dict()
extra_dict = dict(SESSION_ID=utils.get_session_id())

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