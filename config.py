""" 
Config file to define all configs, to change configs, change the input
dictionary for the Classes.
"""

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

	CELL_FIRE_RATE = 1
	CA_STEP_RANGE = [50, 60]
	ENV_SIZE = 0
	LIVING_MASK = True

	def __init__(self, params):
		super().__init__(params)


class _Dataset(_ConfigModule):
	""" Parameters for the Dataset configuration, was previously (pre 25.01.2021)"""
	
	# String determening the dataset class
	NAME = "EMOJI"

	# tuple defining the target emoji(s), possible targets in utils.emoji_dict
	TARGETS = ("lizard", "heart")

	# Defining how the seed will look like: [empty, edges, noise]
	# TODO might be better to add noise as a boolean option?
	# As concatenated string, e.g.: "EDGES NOISE", "EDGES", "EMPTY NOISE"?
	SEED = "EDGES"
	NOISE = 0.4			# Variance of the gaussian noise with 0 mean. 0 for no noise
	# TODO better way of storing?
	ONLY_POS_NOISE = False
	CLIP_NOISE = False

	DAMAGE = 0			# How much of the Dataset will be damaged, 0 for no damage

	# Size of each part of the dataset
	TRAIN_SIZE = 1000
	VAL_SIZE = 100		# TODO maybe it makes more sense to set the size as per part
	# such that each part in the val set will be there, e.g. 10 times or 5 times
	TEST_SIZE = 200

	# Size of the emoji/face, without boundaries
	TARGET_SIZE = 32

	# Size of the full grid
	GRID_SIZE = 40

	def __init__(self, params={}):
		super().__init__(params)
		self.post_process()

	def post_process(self):
		""" checking for correct configuration of the config file """
		assert self.NAME in ["EMOJI", "FACES"]

		if self.NAME == "EMOJI":
			# TODO, post processing does not make sense here, as I will check again in datasets
			# and I need to get the dataset emoji list here. Either define it here, which would
			# not make sense, or check it during dataset...
			pass


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
	RESET_GATE = False 	#  like the update gate, but applied later from seg paper
	RESET_ACTIVATION = None # possible strings for activation functions: ["clipping", "sigmoid"]
	RESET_NOISE = 1e-3

	FLOATX = 'float32' # or float64, which float based will be used for all operations

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
	POOL_SIZE = BATCH_SIZE * 10	# has to be a multiplicative of batch size
	FIXED_SEED_RATIO = 1/8		# Fixing the seed ratio to float: (0,1] 

	POOL_TANH = False # Force pool values between (-1,1)

	""" Ratio of how long to initialize a warm up until the actual ratios kick in
		0 for no warm up, otherwise value between [0,1] How much of total steps """
	WARM_UP = 0. # 0.05 is decent other value, but still shit

	USE_PATTERN_POOL =  False	# 26.01. changed to False, for now not supported!
	MUTATE_POOL = False 		# 12.11 changed from True to False

	USE_Y_POOL = False	# Added option at 16.11

	LOSS_TYPE = "l2"
	ADD_NOISE = True # TODO YOS THIS IS USED DURING EACH CALL
	LR = 1e-3
	LAYER_NORM = True

	REGULARIZER = False 
	FORCE_ALIVE = False 	# Forcing the middle cells to have alpha >= 0.1

	# works instead of the pool, prob to reuse output for next in, 0 to disable
	KEEP_OUTPUT = 0.5 		# Changed standard to 0.5 from 0 on 26.01.2021

	def __init__(self, params={}):
		super().__init__(params)
		self.post_process()

	def post_process(self):
		pass


class _Extra(_ConfigModule):
	""" Extra configuraiots, not fittign other modules """
	USE_TIMER = False	# changed on 05.01.2021 as never used
	LOG_PATH = "logs/maybe/"
	SESSION_ID = None
	PRINT_LAYER = False
	LIST_OF_WEIGHT_NAMES = "all" # List containing the string names of weights to plot in tensorboard, 
		# "all" for every weight
	TENSORBOARD = False
	TB_GRAPH = False
	LOG_LAYERS = False # Logs output of layers (ugly), only works in eager mode
	FIXED_SEED = 0	# fixed seed, 0 for no fixed seed

	VAL_LOG_INTERVALL = 50		# How often to update validation loss (everey X steps)
	
	def __init__(self, params={}):
		super().__init__(params)


# Calls and creates all config objects, to be called after importing the module
# Session Id is set here to make it easier to recognize plots as they all have the id now.
world_dict = dict()
data_dict  = dict(NAME="EMOJI", TARGETS=50, CLIP_NOISE=True)
model_dict = dict()
train_dict = dict(BATCH_SIZE=16)
extra_dict = dict(SESSION_ID=utils.get_session_id())

WORLD = _World(world_dict)
DATA  = _Dataset(data_dict)
MODEL = _Model(model_dict)
TRAIN = _Training(train_dict)
EXTRA = _Extra(extra_dict)

# This is used to save and load the classes
ALL_CONFIG_CLASSES = ["WORLD", "DATA", "MODEL", "TRAIN", "EXTRA"]
