"""
Testing module to use ca and stuff from trainer
"""
import numpy as np
import textwrap
import utils
import display as disp
import config as cfg


def plot_loss_over_time(x, y, ca, loss, number_of_steps):

	# tf.config.run_functions_eagerly(True)
	# cfg.EXTRA.PRINT_LAYER = True

	loss_arr = []
	for i in range(number_of_steps):
		cur_loss = loss(x,y)
		loss_arr.append(cur_loss)
		x = ca(x)

	# fig = disp.plot_loss(np.array(loss_arr)[:,0], plot_mean=False, return_plot=True)
	  
	fig = disp.make_small_fig(y=np.array(np.array(loss_arr)[:,0]), return_fig=False)
	# fig.show()
	# vis.plotlyplot(fig)


def plot_min_max_mean_simple(x, ca, steps=200):
	""" plots mmm of every step if the output of the ca """
	log = {
	    "max":[],
	    "min":[],
	    "mean":[]}

	for i in range(steps):
		log["max"].append(np.amax(x))
		log["min"].append(np.amin(x))
		log["mean"].append(np.mean(x))
		x = ca(x)

	disp.plot_min_max_mean(log).show()


def get_min_max_mean_all_layers(x, ca, steps=200):

	assert cfg.EXTRA.LOG_LAYERS, "only works if 'cfg.EXTRA.LOG_LAYERS' is enabled"

	log = {
	"cnn":{
	    "max":[],
	    "min":[],
	    "mean":[]},
	"in":{
	    "max":[],
	    "min":[],
	    "mean":[]},
	"out":{
	    "max":[],
	    "min":[],
	    "mean":[]}
	}

	for i in range(steps):
		x, cnn_l = ca(x)

		log["out"]["max"].append(np.amax(x.numpy()))
		log["out"]["min"].append(np.amin(x.numpy()))
		log["out"]["mean"].append(np.mean(x.numpy()))

		log["cnn"]["max"].append(np.amax(cnn_l.numpy()))
		log["cnn"]["min"].append(np.amin(cnn_l.numpy()))
		log["cnn"]["mean"].append(np.mean(cnn_l.numpy()))

	return log


def plot_min_max_mean_all_layers(x, ca, steps=200):
	""" simple plot to plot min max and mean """
	# TODO not tested

	log = get_min_max_mean_all_layers(x, ca, steps)
	disp.plot_min_max_mean(log["cnn"], "cnn").show()
	disp.plot_min_max_mean(log["out"], "out").show()


def test_full_step(step_counter):
	"""
	Fully customizable Test function to test/change every thing inside
	might be outdated at the time of usage!
	"""
	# only difference should be that this function includes the loss log as output!

	raise NotImplementedError("should not be needed, can be removed if not used for long time")

	x0, y0 = trainer.get_new_x_y()

	# trainer.x.assign(x0)
	# trainer.y.assign(y0)

	trainer.num_ca_steps.assign(trainer.get_num_ca_steps())


	x = x0
	x_log = [x]
	loss_log = [trainer.batch_l2_loss(x,trainer.y_train[0:8]).numpy()]
	with tf.GradientTape() as g:
		for i in tf.range(trainer.num_ca_steps):
			x = trainer.ca(x)
			x_log.append(x)
			loss_log.append(trainer.batch_l2_loss(x,trainer.y_train[0:8]).numpy())
		full_loss = trainer.batch_l2_loss(x,y0)
		# print(full_loss)

	grads = g.gradient(full_loss, trainer.ca.trainable_variables)
	# print(grads)
	normed_grads = []
	for layer_grad in grads:
	# If layer has only one value, don't apply norm
	
		normed_grads.append(layer_grad / (tf.norm(layer_grad) + 1e-8))

	grads = normed_grads
	# grads = [g/(tf.norm(g)+1e-8) for g in grads]

	trainer.optimizer.apply_gradients(zip(grads, trainer.ca.trainable_variables))

	trainer.post_train_step(x, y0, step_counter)

	return x_log, full_loss, grads, loss_log


# From initial Code, not used by me so far.
# @title Evaluation functions
def eval_perform_steps(ca, x, yt, num_steps):
	yt_label = tf.argmax(yt, axis=-1)

	live_mask = x[..., 0] > 0.1
	live_mask_fl = tf.expand_dims(tf.cast(live_mask, tf.float32), -1)
	dead_channel = tf.cast(x[..., :1] <= 0.1, tf.float32)

	# for now the metric is aggregating everything.
	total_count = tf.reduce_sum(tf.cast(live_mask, tf.float32))

	avg_accuracy_list = []
	avg_total_agreement_list = []
	for _ in range(1, num_steps + 1):
		x = ca(x)

		y = ca.classify(x)
		y_label = tf.argmax(y, axis=-1)

		correct = tf.equal(y_label,  yt_label) & live_mask
		total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
		avg_accuracy_list.append((total_correct/total_count * 100).numpy().item())

		# agreement metrics
		# Important to exclude dead cells:
		y = y * live_mask_fl
		y_label_plus_mask = tf.argmax(tf.concat([y, dead_channel], -1), axis=-1)
		all_counts = []
		for idx in range(10):
			count_i = tf.reduce_sum(
					tf.cast(tf.equal(y_label_plus_mask, idx), tf.int32), axis=[1,2])
			all_counts.append(count_i)
		all_counts_t = tf.stack(all_counts, 1)
		# Now the trick is that if there is a total agreement, their sum is the same
		# as their max.
		equality = tf.equal(tf.reduce_max(all_counts_t, axis=1),
												tf.reduce_sum(all_counts_t, axis=1))
		sum_agreement = tf.reduce_sum(tf.cast(equality, tf.float32))
		avg_total_agreement_list.append(sum_agreement.numpy().item() / y.shape[0] * 100)

	return avg_accuracy_list, avg_total_agreement_list

def eval_batch_fn(ca, x_test, y_test, num_steps, mutate):
	x = ca.initialize(x_test)
	yt = y_test

	avg_acc_l_1, avg_tot_agr_l_1 = eval_perform_steps(ca, x, yt, num_steps)
	if not mutate:
		return avg_acc_l_1, avg_tot_agr_l_1
	# Accuracy after mutation!
	new_idx = np.random.randint(0, x_test.shape[0]-1, size=x_test.shape[0])
	new_x, yt = x_test[new_idx], y_test[new_idx]
	new_x = tf.reshape(new_x, [-1, 28, 28, 1])
	mutate_mask = tf.cast(new_x > 0.1, tf.float32)

	x = tf.concat([new_x, x[:,:,:,1:] * mutate_mask], -1)

	avg_acc_l_2, avg_tot_agr_l_2 = eval_perform_steps(ca, x, yt, num_steps)

	return avg_acc_l_1 + avg_acc_l_2, avg_tot_agr_l_1 + avg_tot_agr_l_2
