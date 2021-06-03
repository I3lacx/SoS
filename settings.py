""" Settings file, for pre defined configs """

# Store all settings by name
world_settings = {}
data_settings  = {}
model_settings = {}
train_settings = {}
extra_settings = {}

def add_setting(name, world, data, model, train, extra):
	""" globally modifies settings by adding this nammed config """
	if name in list(world_settings.keys()):
		# print(f"{name}, already in world settings list : ", list(world_settings.keys()))
		# print("Overwriting! ...")
		# TODO if reloading the modules, this will fail!
		pass

	world_settings[name] = world
	data_settings[name]  = data
	model_settings[name] = model
	train_settings[name] = train
	extra_settings[name] = extra

# Define targets here will be called when using the emoji dataset
def _create_list_of_targets(category, train_emojis, val_emojis, train_fonts, val_fonts):
	train_targets = []
	for font in train_fonts:
		for emoji in train_emojis:
			train_targets.append("emojis/" + category + "/" + font + "/" + emoji)

	val_targets = []
	for font in val_fonts:
		for emoji in val_emojis:
			val_targets.append("emojis/" + category + "/" + font + "/" + emoji)

	# TODO test targets not defined yet...
	return {"train": train_targets, "val": val_targets, "test": val_targets}

def _get_env_size(targets):
	""" Merge all targets and return set of targets """
	full_set = set()
	for val in targets.values():
		full_set.union(set(val))
	return len(full_set)



""" -------- TARGET LISTS BY NAME -------- """
# possible fonts ordered by amount:
# ['joypixels', 'google', 'apple', 'whatsapp', 'openmoji', 'twitter',
#  'samsung', 'microsoft', 'facebook', 'emojidex', 'lg', 'htc', 'mozilla',
# 'docomo', 'softbank', 'au-kddi', 'emojipedia']
target_lists = {}

# Simple Faces
category = "people"
train_e = ["grinning-face_1f600", "angry-face_1f620", "grinning-face-with-big-eyes_1f603",
"anguished-face_1f627", "anguished-face_1f627", "anguished-face_1f627"]
val_e = train_e
train_f = ['joypixels', 'google', 'apple', 'whatsapp', 'openmoji']
val_f = ['samsung', 'twitter']

target_lists["simple_faces"] = (category, train_e, val_e, train_f, val_f)

# Microsotf a lot of simple Faces, removed weirdly colored (not full yellow) and not faces but persons
category = "people"
train_e = ['face-with-pleading-eyes_1f97a', 'yawning-face_1f971', 'grimacing-face_1f62c', 'white-frowning-face_2639', 'face-with-finger-covering-closed-lips_1f92b', 'face-with-medical-mask_1f637', 'shocked-face-with-exploding-head_1f92f', 'face-screaming-in-fear_1f631', 'grinning-cat-face-with-smiling-eyes_1f638', 'face-with-party-horn-and-party-hat_1f973', 'nerd-face_1f913', 'smiling-face-with-open-mouth-and-cold-sweat_1f605', 'zipper-mouth-face_1f910', 'smiling-cat-face-with-heart-shaped-eyes_1f63b', 'face-throwing-a-kiss_1f618', 'angry-face_1f620', 'smiling-face-with-smiling-eyes-and-three-hearts_1f970', 'smiling-face-with-halo_1f607', 'face-with-open-mouth-vomiting_1f92e', 'white-smiling-face_263a', 'smiling-face-with-sunglasses_1f60e', 'face-with-tears-of-joy_1f602', 'weary-face_1f629', 'pensive-face_1f614', 'cat-face-with-tears-of-joy_1f639', 'confused-face_1f615', 'tired-face_1f62b', 'face-with-stuck-out-tongue-and-tightly-closed-eyes_1f61d', 'face-with-look-of-triumph_1f624', 'kissing-face-with-closed-eyes_1f61a', 'smiling-face-with-smiling-eyes-and-hand-covering-mouth_1f92d', 'face-with-cold-sweat_1f613', 'hugging-face_1f917', 'unamused-face_1f612', 'persevering-face_1f623', 'kissing-face-with-smiling-eyes_1f619', 'face-with-thermometer_1f912', 'expressionless-face_1f611', 'upside-down-face_1f643','smiling-face-with-heart-shaped-eyes_1f60d', 'slightly-frowning-face_1f641', 'disappointed-face_1f61e', 'frowning-face-with-open-mouth_1f626', 'face-with-cowboy-hat_1f920', 'smiling-face-with-open-mouth_1f603', 'smiling-face-with-open-mouth-and-tightly-closed-eyes_1f606', 'crying-face_1f622', 'pouting-cat-face_1f63e', 'smiling-face-with-smiling-eyes_1f60a', 'face-with-open-mouth-and-cold-sweat_1f630', 'money-mouth-face_1f911', 'face-without-mouth_1f636', 'smiling-cat-face-with-open-mouth_1f63a', 'face-with-rolling-eyes_1f644', 'face-with-stuck-out-tongue-and-winking-eye_1f61c', 'sneezing-face_1f927', 'hushed-face_1f62f', 'grinning-face-with-smiling-eyes_1f601', 'grinning-face_1f600', 'flushed-face_1f633', 'slightly-smiling-face_1f642', 'face-with-head-bandage_1f915', 'winking-face_1f609', 'relieved-face_1f60c', 'weary-cat-face_1f640', 'astonished-face_1f632', 'lying-face_1f925', 'face-with-open-mouth_1f62e', 'kissing-face_1f617', 'disappointed-but-relieved-face_1f625', 'face-with-uneven-eyes-and-wavy-mouth_1f974', 'thinking-face_1f914', 'crying-cat-face_1f63f', 'dizzy-face_1f635', 'kissing-cat-face-with-closed-eyes_1f63d', 'drooling-face_1f924', 'smiling-face-with-open-mouth-and-smiling-eyes_1f604', 'face-with-stuck-out-tongue_1f61b', 'sleeping-face_1f634', 'loudly-crying-face_1f62d', 'cat-face-with-wry-smile_1f63c', 'grinning-face-with-star-eyes_1f929', 'worried-face_1f61f', 'sleepy-face_1f62a', 'anguished-face_1f627', 'fearful-face_1f628', 'grinning-face-with-one-large-and-one-small-eye_1f92a', 'face-with-one-eyebrow-raised_1f928']
val_e = ['face-savouring-delicious-food_1f60b', 'face-palm_1f926', 'smirking-face_1f60f', 'face-with-monocle_1f9d0', 'confounded-face_1f616', 'neutral-face_1f610']
train_f = ["microsoft"]
val_f = train_f

target_lists["microsoft_lots_faces"] = (category, train_e, val_e, train_f, val_f)
# Etc...

# Experiment 1 -> low number of samples, easy and complex faces
category = "people"
train_e = ["grinning-face-with-smiling-eyes_1f604", "grinning-face-with-sweat_1f605", "angry-face_1f620", "astonished-face_1f632", 
"face-screaming-in-fear_1f631", "face-with-tears-of-joy_1f602", "beaming-face-with-smiling-eyes_1f601",
]
val_e = train_e + ["downcast-face-with-sweat_1f613", "confused-face_1f615"]

train_f = ['apple', 'joypixels', 'google', 'samsung']
val_f = ['whatsapp', 'twitter']

target_lists["exp_1_faces"] = (category, train_e, val_e, train_f, val_f)


# Experiment 1 -> low number of samples, easy and complex faces
category = "people"
train_e = ["grinning-face-with-smiling-eyes_1f604", "grinning-face-with-sweat_1f605", "angry-face_1f620", "astonished-face_1f632", 
"face-screaming-in-fear_1f631", "face-with-tears-of-joy_1f602", "beaming-face-with-smiling-eyes_1f601",
]
val_e = train_e + ["downcast-face-with-sweat_1f613", "confused-face_1f615"]

train_f = ['apple']
val_f = ['whatsapp']

target_lists["exp_1_faces_single_font"] = (category, train_e, val_e, train_f, val_f)


# Single Emoji Experiment
category = "people"
train_e = ["beaming-face-with-smiling-eyes_1f601"]
val_e = train_e

train_f = ['apple']
val_f = train_f

target_lists["single_emoji"] = (category, train_e, val_e, train_f, val_f)


# TODO is this really right? this should be loadad from external txt file...
import os
# All people single font
category = "people"
# joined lists of both directorys
train_e = list(set(os.listdir("emojis/people/apple")).intersection(set(os.listdir("emojis/people/whatsapp"))))
train_e = [it.split(".")[0] for it in train_e]
val_e = train_e

train_f = ["apple"]
val_f = ["whatsapp"]

target_lists["all_people_single_font"] = (category, train_e, val_e, train_f, val_f)


# MICROSOFT NATURE BIG
category = "nature"

# full list with the first 70 emojis, removing the png
full_list = [it.split(".")[0] for it in os.listdir("emojis/nature/microsoft")][:70]
train_e = full_list[:50]
val_e = full_list[50:]

train_f = ["microsoft"]
val_f = train_f

target_lists["microsoft_nature_big"] = (category, train_e, val_e, train_f, val_f)

# Microsotf a lot of simple Faces, removed weirdly colored (not full yellow) and not faces but persons
category = "people"
train_e = ['face-with-pleading-eyes_1f97a', 'yawning-face_1f971', 'grimacing-face_1f62c', 'white-frowning-face_2639', 'face-with-finger-covering-closed-lips_1f92b', 'face-with-medical-mask_1f637', 'shocked-face-with-exploding-head_1f92f', 'face-screaming-in-fear_1f631', 'grinning-cat-face-with-smiling-eyes_1f638', 'face-with-party-horn-and-party-hat_1f973', 'nerd-face_1f913', 'smiling-face-with-open-mouth-and-cold-sweat_1f605', 'zipper-mouth-face_1f910', 'smiling-cat-face-with-heart-shaped-eyes_1f63b', 'face-throwing-a-kiss_1f618', 'angry-face_1f620', 'smiling-face-with-smiling-eyes-and-three-hearts_1f970', 'smiling-face-with-halo_1f607', 'face-with-open-mouth-vomiting_1f92e', 'white-smiling-face_263a', 'smiling-face-with-sunglasses_1f60e', 'face-with-tears-of-joy_1f602', 'weary-face_1f629', 'pensive-face_1f614', 'cat-face-with-tears-of-joy_1f639', 'confused-face_1f615', 'tired-face_1f62b', 'face-with-stuck-out-tongue-and-tightly-closed-eyes_1f61d', 'face-with-look-of-triumph_1f624', 'kissing-face-with-closed-eyes_1f61a', 'smiling-face-with-smiling-eyes-and-hand-covering-mouth_1f92d', 'face-with-cold-sweat_1f613', 'hugging-face_1f917', 'unamused-face_1f612', 'persevering-face_1f623', 'kissing-face-with-smiling-eyes_1f619', 'face-with-thermometer_1f912', 'expressionless-face_1f611', 'upside-down-face_1f643','smiling-face-with-heart-shaped-eyes_1f60d', 'slightly-frowning-face_1f641', 'disappointed-face_1f61e', 'frowning-face-with-open-mouth_1f626', 'face-with-cowboy-hat_1f920', 'smiling-face-with-open-mouth_1f603', 'smiling-face-with-open-mouth-and-tightly-closed-eyes_1f606', 'crying-face_1f622', 'pouting-cat-face_1f63e', 'smiling-face-with-smiling-eyes_1f60a', 'face-with-open-mouth-and-cold-sweat_1f630']
val_e = ['face-savouring-delicious-food_1f60b', 'face-palm_1f926', 'smirking-face_1f60f', 'face-with-monocle_1f9d0', 'confounded-face_1f616', 'neutral-face_1f610', 'kissing-cat-face-with-closed-eyes_1f63d', 'drooling-face_1f924', 'smiling-face-with-open-mouth-and-smiling-eyes_1f604', 'face-with-stuck-out-tongue_1f61b', 'sleeping-face_1f634', 'loudly-crying-face_1f62d', 'cat-face-with-wry-smile_1f63c', 'grinning-face-with-star-eyes_1f929', 'worried-face_1f61f', 'sleepy-face_1f62a', 'anguished-face_1f627', 'fearful-face_1f628', 'grinning-face-with-one-large-and-one-small-eye_1f92a', 'face-with-one-eyebrow-raised_1f928']
# 24 removed from train
# 'money-mouth-face_1f911', 'face-without-mouth_1f636', 'smiling-cat-face-with-open-mouth_1f63a', 'face-with-rolling-eyes_1f644', 'face-with-stuck-out-tongue-and-winking-eye_1f61c', 'sneezing-face_1f927', 'hushed-face_1f62f', 'grinning-face-with-smiling-eyes_1f601', 'grinning-face_1f600', 'flushed-face_1f633', 'slightly-smiling-face_1f642', 'face-with-head-bandage_1f915', 'winking-face_1f609', 'relieved-face_1f60c', 'weary-cat-face_1f640', 'astonished-face_1f632', 'lying-face_1f925', 'face-with-open-mouth_1f62e', 'kissing-face_1f617', 'disappointed-but-relieved-face_1f625', 'face-with-uneven-eyes-and-wavy-mouth_1f974', 'thinking-face_1f914', 'crying-cat-face_1f63f', 'dizzy-face_1f635'
train_f = ["microsoft"]
val_f = train_f

target_lists["microsoft_50_faces"] = (category, train_e, val_e, train_f, val_f)


# Openmoji Images, can be used for validation on microsoft with same emojis
category = "people"
train_e = ["pleading-face_1f97a", "face-screaming-in-fear_1f631", "grinning-cat-with-smiling-eyes_1f638"]
val_e = ["smirking-face_1f60f", "cat-with-wry-smile_1f63c"]

# Not there:  "face-palm_1f926",
train_f = ["openmoji"]
val_f = train_f

target_lists["openmoji_faces"] = (category, train_e, val_e, train_f, val_f)


# MICROSOFT FACES VERY BIG
category = "people"

# full list with the first 70 emojis, removing the png
full_list = [it.split(".")[0] for it in os.listdir("emojis/people/microsoft")][:400]
train_e = full_list[:300]
val_e = full_list[300:]

train_f = ["microsoft"]
val_f = train_f

target_lists["microsoft_people_300"] = (category, train_e, val_e, train_f, val_f)
""" ----------- DEFINING THE SETTINGS BY NAME -------- """ 

# EMOJI_DATA_STD (created: 01.03.2021)
# Simple faces with hidden information, no noise
_targets = _create_list_of_targets(*target_lists["simple_faces"])
world = dict(ENV_SIZE=_get_env_size(_targets))
data  = dict(NOISE=0., NAME="EMOJI_DATA", BINARY_X=True, CLIP_NOISE=True,
 BINARY_NOISE=True, TARGETS=_targets)
model = dict()
train = dict(BATCH_SIZE=16)
extra = dict()
add_setting("emoji_data_std", world, data, model, train, extra)

# EMOJI_DATA_NO_ENV (created: 02.03.2021)
# Simple faces without hidden information, no noise
_targets = _create_list_of_targets(*target_lists["simple_faces"])
world = dict()
data  = dict(NOISE=0., NAME="EMOJI_DATA", BINARY_X=True, CLIP_NOISE=True, BINARY_NOISE=True, TARGETS=_targets)
model = dict()
train = dict(BATCH_SIZE=16)
extra = dict()
add_setting("emoji_data_no_env", world, data, model, train, extra)


_targets = _create_list_of_targets(*target_lists["microsoft_lots_faces"])
world = dict()
data  = dict(NOISE=0., NAME="EMOJI_DATA", BINARY_X=True, CLIP_NOISE=True,
 BINARY_NOISE=True, TARGETS=_targets)
model = dict()
train = dict(BATCH_SIZE=16)
extra = dict()
add_setting("emoji_data_microsoft_faces", world, data, model, train, extra)


_targets = _create_list_of_targets(*target_lists["microsoft_lots_faces"])
world = dict()
data  = dict(NOISE=0.4, NAME="EMOJI_DATA", BINARY_X=True, CLIP_NOISE=True, BINARY_NOISE=True, TARGETS=_targets)
model = dict()
train = dict(BATCH_SIZE=16)
extra = dict()
add_setting("emoji_data_microsoft_faces_w_noise", world, data, model, train, extra)

# Faces
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="FACES", TARGETS=5)
model_dict = dict()
train_dict = dict()
extra_dict = dict()
add_setting("simple_faces", world_dict, data_dict, model_dict, train_dict, extra_dict)


img_path = "emojis/people/microsoft/grinning-face-with-smiling-eyes_1f601"
_targets = {"train": [img_path], "val":[img_path], "test":[img_path]}
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, SEED="BLANK")
model_dict = dict()
train_dict = dict(BATCH_SIZE=8)
extra_dict = dict()
add_setting("single_from_blank", world_dict, data_dict, model_dict, train_dict, extra_dict)

# Experiment 1 settings (created: 09.03.2021)
# Simple faces w/o hidden information, no noise
_targets = _create_list_of_targets(*target_lists["exp_1_faces"])
world = dict()
data  = dict(NOISE=0., NAME="EMOJI_DATA", BINARY_X=True, CLIP_NOISE=True,
 BINARY_NOISE=True, TARGETS=_targets)
model = dict()
train = dict(BATCH_SIZE=16)
extra = dict()
add_setting("exp_1_faces", world, data, model, train, extra)

# Experiment 1_2 settings (created: 09.03.2021)
# Simple faces w/o hidden information, no noise, different edges
_targets = _create_list_of_targets(*target_lists["exp_1_faces"])
world = dict()
data  = dict(NOISE=0., NAME="EMOJI_DATA", BINARY_X=True,
 CLIP_NOISE=True, BINARY_NOISE=True, TARGETS=_targets,
 EDGE_DETECTION="ADAPTIVE_GAUSSIAN")
model = dict()
train = dict(BATCH_SIZE=16)
extra = dict()
add_setting("exp_1_faces_gaus_edges", world, data, model, train, extra)

_targets = _create_list_of_targets(*target_lists["exp_1_faces_single_font"])
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", BINARY_X=True,
 TARGETS=_targets, EDGE_DETECTION="ADAPTIVE_GAUSSIAN",
 GANCA_NORM_INPUT=False, GRID_SIZE=80, TARGET_SIZE=64, DISC_INPUT_NOISE=True)
model_dict = dict(NAME="GANCA", LEAKY_RELU=False,
 CHANNEL_N=16, 	GANCA_TANH=True, GANCA_EXTRA_LAYER=False, HIDDEN_LAYERS=1)
train_dict = dict(BATCH_SIZE=16)
extra_dict = dict()
add_setting("high_res_ganca_working_ish", world_dict, data_dict, model_dict, train_dict, extra_dict)

_targets = _create_list_of_targets(*target_lists["microsoft_lots_faces"])
world_dict = dict(CA_STEP_RANGE=[50,60])
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", BINARY_X=True,
 TARGETS=_targets, EDGE_DETECTION="ADAPTIVE_GAUSSIAN",
 GANCA_NORM_INPUT=False, GRID_SIZE=40, TARGET_SIZE=32, DISC_INPUT_NOISE=False, GANCA_GEN_NOISE=False)
model_dict = dict(NAME="GANCA", LEAKY_RELU=False, DISC_SIZE=128,
 CHANNEL_N=16, 	GANCA_TANH=False, GANCA_EXTRA_LAYER=False, HIDDEN_LAYERS=0,
 GEN_L2_LR=1e-3, GANCA_LR=1e-3, DISC_LR=1e-4, LATENT_DIM=100)
train_dict = dict(BATCH_SIZE=16, LABEL_SMOOTHING=0, GANCA_USE_OLD=False, GANCA_LAYER_NORM=True,
	GEN_L2_LOSS=False, GANCA_LOSS_TYPE="WGAN")
extra_dict = dict()
add_setting("wanca_w_latent", world_dict, data_dict, model_dict, train_dict, extra_dict)


# Single Emoiji testing basic performance, changing hidden_filter, channel_n and hidden_layers to test performance
_targets = _create_list_of_targets(*target_lists["single_emoji"])
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, EDGE_DETECTION="CANNY")
model_dict = dict(HIDDEN_FILTER_SIZE=128, CHANNEL_N=16, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=8)
extra_dict = dict()
add_setting("experiment_basic_nca_init", world_dict, data_dict, model_dict, train_dict, extra_dict)


# Single Emoiji testing basic performance, changing hidden_filter, channel_n and hidden_layers to test performance
_targets = _create_list_of_targets(*target_lists["microsoft_lots_faces"])
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, EDGE_DETECTION="CANNY")
model_dict = dict(HIDDEN_FILTER_SIZE=128, CHANNEL_N=16, HIDDEN_LAYERS=0,
	NCA_ACTIVATION="RATIONAL")
train_dict = dict(BATCH_SIZE=16)
extra_dict = dict()
add_setting("rationals_testing", world_dict, data_dict, model_dict, train_dict, extra_dict)


_targets = {"train": ["wsr_i"], "val":["wsr_i"], "test":["wsr_i"]}
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, SEED="BLANK")
model_dict = dict(HIDDEN_FILTER_SIZE=128, CHANNEL_N=16, HIDDEN_LAYERS=0,
	NCA_ACTIVATION="RELU")
train_dict = dict(BATCH_SIZE=16)
extra_dict = dict()
add_setting("wsr", world_dict, data_dict, model_dict, train_dict, extra_dict)


# ----THESIS---- microsoft nature 50+ emojis experiment:
_targets = _create_list_of_targets(*target_lists["microsoft_nature_big"])
world_dict = dict(CA_STEP_RANGE=[50,60])
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets,
 EDGE_DETECTION="CANNY", GRID_SIZE=40, TARGET_SIZE=32)
model_dict = dict(NAME="NCA", NCA_ACTIVATION="RELU", CHANNEL_N=16, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16)
extra_dict = dict()
add_setting("exp_nature_50", world_dict, data_dict, model_dict, train_dict, extra_dict)

# --THESIS:--- microsoft 50 faces emojis experiment:
_targets = _create_list_of_targets(*target_lists["microsoft_50_faces"])
world_dict = dict(CA_STEP_RANGE=[50,60])
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets,
 EDGE_DETECTION="CANNY", GRID_SIZE=40, TARGET_SIZE=32)
model_dict = dict(NAME="NCA", NCA_ACTIVATION="RELU", CHANNEL_N=16, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16)
extra_dict = dict()
add_setting("exp_faces_50", world_dict, data_dict, model_dict, train_dict, extra_dict)

# --THESIS-- microsoft 50 faces emojis with superpixel information experiment:
# Changable Superpixel_num_segments value!
_targets = _create_list_of_targets(*target_lists["openmoji_faces"])
# _targets = _create_list_of_targets(*target_lists["microsoft_50_faces"])
world_dict = dict(ENV_SIZE=4)
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, SUPERPIXEL_FIXED=True,
 EDGE_DETECTION="CANNY", GRID_SIZE=40, TARGET_SIZE=40, SUPERPIXEL_NUM_SEGMENTS=10)
model_dict = dict(NAME="NCA", NCA_ACTIVATION="RELU", CHANNEL_N=16, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16)
extra_dict = dict()
add_setting("exp_faces_50_super", world_dict, data_dict, model_dict, train_dict, extra_dict)


# --THESIS:--- microsoft 50 faces emojis experiment:
_targets = _create_list_of_targets(*target_lists["openmoji_faces"])
world_dict = dict(CA_STEP_RANGE=[50,60])
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets,
 EDGE_DETECTION="CANNY", GRID_SIZE=40, TARGET_SIZE=40)
model_dict = dict(NAME="NCA", NCA_ACTIVATION="RELU", CHANNEL_N=16, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16)
extra_dict = dict()
add_setting("good_nca_openmoji_data", world_dict, data_dict, model_dict, train_dict, extra_dict)


# microsoft 50 faces emojis experiment:
_targets = _create_list_of_targets(*target_lists["microsoft_50_faces"])
world_dict = dict(CA_STEP_RANGE=[50,60])
data_dict  = dict(NOISE=0.3, NAME="EMOJI_DATA", TARGETS=_targets,
 EDGE_DETECTION="CANNY", GRID_SIZE=40, TARGET_SIZE=32)
model_dict = dict(NAME="NCA", NCA_ACTIVATION="RELU", CHANNEL_N=16, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16)
extra_dict = dict()
add_setting("exp_faces_50_noise", world_dict, data_dict, model_dict, train_dict, extra_dict)


# TODO this should be the 100 not 88
# microsoft 50 faces emojis experiment:
_targets = _create_list_of_targets(*target_lists["microsoft_lots_faces"])
world_dict = dict(CA_STEP_RANGE=[50,60])
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets,
 EDGE_DETECTION="CANNY", GRID_SIZE=40, TARGET_SIZE=32)
model_dict = dict(NAME="NCA", NCA_ACTIVATION="RELU", CHANNEL_N=16, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16)
extra_dict = dict()
add_setting("exp_faces_88", world_dict, data_dict, model_dict, train_dict, extra_dict)

# TODO wtf don't be working too
# microsoft 50 faces emojis experiment:
_targets = _create_list_of_targets(*target_lists["microsoft_people_300"])
world_dict = dict(CA_STEP_RANGE=[50,60])
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets,
 EDGE_DETECTION="CANNY", GRID_SIZE=40, TARGET_SIZE=32)
model_dict = dict(NAME="NCA", NCA_ACTIVATION="RELU", CHANNEL_N=16, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16)
extra_dict = dict()
add_setting("exp_people_300", world_dict, data_dict, model_dict, train_dict, extra_dict)

# TODO wtf don't be working too
# microsoft 50 faces emojis experiment:
_targets = _create_list_of_targets(*target_lists["microsoft_people_300"])
world_dict = dict(ENV_SIZE=4)
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, SUPERPIXEL_FIXED=True,
 EDGE_DETECTION="CANNY", GRID_SIZE=40, TARGET_SIZE=32, SUPERPIXEL_NUM_SEGMENTS=50)
model_dict = dict(NAME="NCA", NCA_ACTIVATION="RELU", CHANNEL_N=16, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16)
extra_dict = dict()
add_setting("exp_people_300_super", world_dict, data_dict, model_dict, train_dict, extra_dict)


# Good ganca -- USED IN THESES!----
_targets = _create_list_of_targets(*target_lists["microsoft_50_faces"])
# _targets = _create_list_of_targets(*target_lists["openmoji_faces"])
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, EDGE_DETECTION="CANNY",
 GANCA_NORM_INPUT=False, GRID_SIZE=40, TARGET_SIZE=32, DISC_INPUT_NOISE=True, GANCA_GEN_NOISE=True)
model_dict = dict(NAME="GANCA", NCA_ACTIVATION="RELU",
 CHANNEL_N=16, 	GANCA_TANH=True, GANCA_EXTRA_LAYER=False, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16, LABEL_SMOOTHING=0.3, GANCA_USE_OLD=True, GANCA_LAYER_NORM=True)
extra_dict = dict()
add_setting("good_ganca", world_dict, data_dict, model_dict, train_dict, extra_dict)

# Good ganca -- USED IN THESES!----
_targets = _create_list_of_targets(*target_lists["microsoft_50_faces"])
# _targets = _create_list_of_targets(*target_lists["openmoji_faces"])
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, EDGE_DETECTION="CANNY",
 GANCA_NORM_INPUT=False, GRID_SIZE=40, TARGET_SIZE=32, DISC_INPUT_NOISE=True, GANCA_GEN_NOISE=True, 
 GANCA_NOISE_STD=0.01)
model_dict = dict(NAME="GANCA", NCA_ACTIVATION="RELU",
 CHANNEL_N=16, 	GANCA_TANH=True, GANCA_EXTRA_LAYER=False, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16, LABEL_SMOOTHING=0.1, GANCA_USE_OLD=True, GANCA_LAYER_NORM=True)
extra_dict = dict()
add_setting("better_ganca", world_dict, data_dict, model_dict, train_dict, extra_dict)


_targets = _create_list_of_targets(*target_lists["microsoft_50_faces"])
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, EDGE_DETECTION="CANNY",
 GANCA_NORM_INPUT=False, GRID_SIZE=40, TARGET_SIZE=32, DISC_INPUT_NOISE=False, GANCA_GEN_NOISE=False)
model_dict = dict(NAME="GANCA", LEAKY_RELU=False,
 CHANNEL_N=16, 	GANCA_TANH=True, GANCA_EXTRA_LAYER=False, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16, LABEL_SMOOTHING=0.3, GANCA_USE_OLD=True, GANCA_LAYER_NORM=True)
extra_dict = dict()
add_setting("ganca_label_smoothing", world_dict, data_dict, model_dict, train_dict, extra_dict)

_targets = _create_list_of_targets(*target_lists["microsoft_50_faces"])
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, EDGE_DETECTION="CANNY",
 GANCA_NORM_INPUT=False, GRID_SIZE=40, TARGET_SIZE=32, DISC_INPUT_NOISE=True, GANCA_GEN_NOISE=True,
 GANCA_NOISE_STD=0.01)
model_dict = dict(NAME="GANCA", LEAKY_RELU=False,
 CHANNEL_N=16, 	GANCA_TANH=True, GANCA_EXTRA_LAYER=False, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16, LABEL_SMOOTHING=0, GANCA_USE_OLD=True, GANCA_LAYER_NORM=True)
extra_dict = dict()
add_setting("ganca_noise", world_dict, data_dict, model_dict, train_dict, extra_dict)

_targets = _create_list_of_targets(*target_lists["microsoft_50_faces"])
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, EDGE_DETECTION="CANNY",
 GANCA_NORM_INPUT=False, GRID_SIZE=40, TARGET_SIZE=32, DISC_INPUT_NOISE=False, GANCA_GEN_NOISE=False)
model_dict = dict(NAME="GANCA", LEAKY_RELU=False,
 CHANNEL_N=16, 	GANCA_TANH=True, GANCA_EXTRA_LAYER=False, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16, LABEL_SMOOTHING=0, GANCA_USE_OLD=True, GANCA_LAYER_NORM=True)
extra_dict = dict()
add_setting("first_ganca", world_dict, data_dict, model_dict, train_dict, extra_dict)

_targets = _create_list_of_targets(*target_lists["microsoft_50_faces"])
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, EDGE_DETECTION="CANNY",
 GANCA_NORM_INPUT=False, GRID_SIZE=40, TARGET_SIZE=32, DISC_INPUT_NOISE=False, GANCA_GEN_NOISE=False)
model_dict = dict(NAME="GANCA", LEAKY_RELU=False,
 CHANNEL_N=16, 	GANCA_TANH=False, GANCA_EXTRA_LAYER=False, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16, LABEL_SMOOTHING=0, GANCA_USE_OLD=False, GANCA_LAYER_NORM=False)
extra_dict = dict()
add_setting("worst_ganca", world_dict, data_dict, model_dict, train_dict, extra_dict)


# Good ganca (testing) This did not work so far:
_targets = _create_list_of_targets(*target_lists["microsoft_people_300"])
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, EDGE_DETECTION="CANNY",
 GANCA_NORM_INPUT=False, GRID_SIZE=40, TARGET_SIZE=32, DISC_INPUT_NOISE=True, GANCA_GEN_NOISE=True)
model_dict = dict(NAME="GANCA", NCA_ACTIVATION="RELU",
 CHANNEL_N=16, 	GANCA_TANH=True, GANCA_EXTRA_LAYER=False, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16, LABEL_SMOOTHING=0.3, GANCA_USE_OLD=True, GANCA_LAYER_NORM=True)
extra_dict = dict()
add_setting("good_ganca_big", world_dict, data_dict, model_dict, train_dict, extra_dict)


# THESIS: good wanca settings used:
_targets = _create_list_of_targets(*target_lists["microsoft_50_faces"])
# _targets = _create_list_of_targets(*target_lists["openmoji_faces"])
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, EDGE_DETECTION="CANNY",
 GANCA_NORM_INPUT=False, GRID_SIZE=40, TARGET_SIZE=32, DISC_INPUT_NOISE=True, GANCA_GEN_NOISE=True)
model_dict = dict(NAME="GANCA", NCA_ACTIVATION="RELU", BUGGED_DISC=True,
 CHANNEL_N=16, 	GANCA_TANH=True, GANCA_EXTRA_LAYER=False, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16, LABEL_SMOOTHING=0.3, GANCA_USE_OLD=True, GANCA_LAYER_NORM=True,
	GANCA_LOSS_TYPE="WGAN")
extra_dict = dict()
add_setting("good_wanca", world_dict, data_dict, model_dict, train_dict, extra_dict)


# VERY BIG WANCA 
_targets = _create_list_of_targets(*target_lists["microsoft_people_300"])
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, EDGE_DETECTION="CANNY",
 GANCA_NORM_INPUT=False, GRID_SIZE=64, TARGET_SIZE=60, DISC_INPUT_NOISE=True, GANCA_GEN_NOISE=True,
 GANCA_NOISE_STD=0.01)
model_dict = dict(NAME="GANCA", NCA_ACTIVATION="RELU", BUGGED_DISC=True, DISC_SIZE=64,
 CHANNEL_N=64, 	GANCA_TANH=True, GANCA_EXTRA_LAYER=False, HIDDEN_LAYERS=1)
train_dict = dict(BATCH_SIZE=8, LABEL_SMOOTHING=0.1, GANCA_USE_OLD=True, GANCA_LAYER_NORM=True,
	GANCA_LOSS_TYPE="WGAN")
extra_dict = dict()
add_setting("very_big_wanca", world_dict, data_dict, model_dict, train_dict, extra_dict)


#ganca w superpixels?????
_targets = _create_list_of_targets(*target_lists["microsoft_50_faces"])
world_dict = dict(ENV_SIZE=4)
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, EDGE_DETECTION="CANNY",
 GANCA_NORM_INPUT=False, GRID_SIZE=40, TARGET_SIZE=32, DISC_INPUT_NOISE=True,SUPERPIXEL_FIXED=True,
  GANCA_GEN_NOISE=True, SUPERPIXEL_NUM_SEGMENTS=10)
model_dict = dict(NAME="GANCA", NCA_ACTIVATION="RELU",
 CHANNEL_N=16, 	GANCA_TANH=True, GANCA_EXTRA_LAYER=False, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16, LABEL_SMOOTHING=0.3, GANCA_USE_OLD=True, GANCA_LAYER_NORM=True)
extra_dict = dict()
add_setting("super_ganca", world_dict, data_dict, model_dict, train_dict, extra_dict)


# microsoft 50 faces emojis super random exp:
_targets = _create_list_of_targets(*target_lists["microsoft_people_300"])
world_dict = dict(CA_STEP_RANGE=[50,60])
data_dict  = dict(NOISE=0.3, NAME="EMOJI_DATA", TARGETS=_targets, SUPERPIXEL_FIXED=True,
 EDGE_DETECTION="CANNY", GRID_SIZE=40, TARGET_SIZE=32, SUPERPIXEL_RANDOMIZED=True)
model_dict = dict(NAME="NCA", NCA_ACTIVATION="RELU", CHANNEL_N=16, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16)
extra_dict = dict()
add_setting("exp_faces_super_random", world_dict, data_dict, model_dict, train_dict, extra_dict)



# Testing/always changing configurations for nca:
env_len = 4
_targets = _create_list_of_targets(*target_lists["openmoji_faces"])
world_dict = dict(CA_STEP_RANGE=[50,60], ENV_SIZE=env_len)
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, SUPERPIXEL_FIXED=True,
 EDGE_DETECTION="CANNY", GRID_SIZE=40, TARGET_SIZE=40, SUPERPIXEL_NUM_SEGMENTS=50)
model_dict = dict(NAME="NCA", NCA_ACTIVATION="RELU", CHANNEL_N=16, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16)
extra_dict = dict()
add_setting("nca_testing", world_dict, data_dict, model_dict, train_dict, extra_dict)


# --- Presentation --- questions
_targets = {"train": ["questions_i2"], "val":["questions_i2"], "test":["questions_i2"]}
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, SEED="BLANK", GRID_SIZE=74, TARGET_SIZE=64)
model_dict = dict(HIDDEN_FILTER_SIZE=128, CHANNEL_N=16, HIDDEN_LAYERS=0,
	NCA_ACTIVATION="RELU")
train_dict = dict(BATCH_SIZE=8)
extra_dict = dict()
add_setting("questions", world_dict, data_dict, model_dict, train_dict, extra_dict)

# Disc training 
_targets = _create_list_of_targets(*target_lists["microsoft_50_faces"])
world_dict = dict()
data_dict  = dict(NOISE=.3, NAME="EMOJI_DATA", TARGETS=_targets, EDGE_DETECTION="CANNY",
 GANCA_NORM_INPUT=False, GRID_SIZE=40, TARGET_SIZE=32, DISC_INPUT_NOISE=True,SUPERPIXEL_FIXED=True,
  GANCA_GEN_NOISE=True, TRAIN_SIZE = 10000)
model_dict = dict(NAME="GANCA", NCA_ACTIVATION="RELU", TRAIN_DISC=True,
 CHANNEL_N=16, 	GANCA_TANH=True, GANCA_EXTRA_LAYER=False, HIDDEN_LAYERS=0)
train_dict = dict(BATCH_SIZE=16, LABEL_SMOOTHING=0.3, GANCA_USE_OLD=True, GANCA_LAYER_NORM=True)
extra_dict = dict()
add_setting("disc_testing", world_dict, data_dict, model_dict, train_dict, extra_dict)

# VERY BIG GANCA 
_targets = _create_list_of_targets(*target_lists["microsoft_people_300"])
world_dict = dict()
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", TARGETS=_targets, EDGE_DETECTION="CANNY",
 GANCA_NORM_INPUT=False, GRID_SIZE=40, TARGET_SIZE=32, DISC_INPUT_NOISE=True, GANCA_GEN_NOISE=True,
 GANCA_NOISE_STD=0.01)
model_dict = dict(NAME="GANCA", NCA_ACTIVATION="RELU",
 CHANNEL_N=64, 	GANCA_TANH=True, GANCA_EXTRA_LAYER=False, HIDDEN_LAYERS=1)
train_dict = dict(BATCH_SIZE=16, LABEL_SMOOTHING=0.1, GANCA_USE_OLD=True, GANCA_LAYER_NORM=True)
extra_dict = dict()
add_setting("very_big_ganca", world_dict, data_dict, model_dict, train_dict, extra_dict)


# Adding self drawn images to training list
_targets = _create_list_of_targets(*target_lists["microsoft_lots_faces"])
self_drawn = os.listdir("self_drawn/canny")
self_drawn_names = ["self_drawn/canny/" + name.split(".")[0] for name in self_drawn]
env_len = len(_targets["train"]) + len(_targets["val"])
# env_len = 0
# _targets["train"] += self_drawn_names
world_dict = dict(CA_STEP_RANGE=[50,60], ENV_SIZE=env_len)
data_dict  = dict(NOISE=0., NAME="EMOJI_DATA", BINARY_X=True,
 TARGETS=_targets, EDGE_DETECTION="CANNY", ENVIRONMENT_INFORMATION="EMOJI",
 GANCA_NORM_INPUT=False, GRID_SIZE=40, TARGET_SIZE=40, DISC_INPUT_NOISE=False, GANCA_GEN_NOISE=False,
 SUPERPIXEL_COLOR=True)
model_dict = dict(NAME="NCA", NCA_ACTIVATION="RELU", DISC_SIZE=128,
 CHANNEL_N=16, 	GANCA_TANH=False, GANCA_EXTRA_LAYER=False, HIDDEN_LAYERS=0,
 GEN_L2_LR=1e-3, GANCA_LR=1e-3, DISC_LR=1e-4, LATENT_DIM=None)
train_dict = dict(BATCH_SIZE=4, LABEL_SMOOTHING=0, GANCA_USE_OLD=False, GANCA_LAYER_NORM=True,
	GEN_L2_LOSS=False, GANCA_LOSS_TYPE="CROSS_ENTROPY", REGULARIZER=False)
extra_dict = dict()
add_setting("ganca_testing", world_dict, data_dict, model_dict, train_dict, extra_dict)
