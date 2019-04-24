import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import *

def get_test_metrics(model_id):
	model_dir = get_model_dir(model_id)

	acc_meter = get_pickle_file(model_dir, 'test_accuracy_meter.p')
	entropy_meter = get_pickle_file(model_dir, 'test_entropy_meter.p')
	distinctness_meter = get_pickle_file(model_dir, 'test_distinctness_meter.p')
	rsa_sr_meter = get_pickle_file(model_dir, 'test_rsa_sr_meter.p')
	rsa_si_meter = get_pickle_file(model_dir, 'test_rsa_si_meter.p')
	rsa_ri_meter = get_pickle_file(model_dir, 'test_rsa_ri_meter.p')
	topological_sim_meter = get_pickle_file(model_dir, 'test_topological_sim_meter.p')

	return (acc_meter.avg,
			entropy_meter.avg,
			distinctness_meter.avg,
			rsa_sr_meter.avg,
			rsa_si_meter.avg,
			rsa_ri_meter.avg,
			topological_sim_meter.avg)

def get_test_messages_stats(model_id, vocab_size, data_folder, plots_dir, should_plot=False):
	_, idx_to_word, padding_idx = load_dictionaries(vocab_size)

	model_dir = get_model_dir(model_id)

	# Load messages
	messages = get_pickle_file(model_dir, 'test_messages.p')

	# Grab stats
	min_len, max_len, avg_len, counter = compute_stats(messages, padding_idx, idx_to_word) # counter includes <S> (aka EOS)

	n_utt = len(counter)

	if should_plot:
		# Plots
		top_common_percent = 1 if vocab_size < 50 else 0.3
		plot_token_frequency(counter, n_utt, top_common_percent, model_id, plots_dir)
		plot_token_distribution(counter, model_id, plots_dir)

		# Token correlation with features
		metadata = pickle.load(open('../shapes/{}/test.metadata.p'.format(data_folder), 'rb'))

		token_to_attr, attr_to_token = get_attributes_dicts(messages, metadata, idx_to_word)

		#plot_attributes_per_token(counter, n_utt, top_common_percent, token_to_attr, model_id, plots_dir)
		n_top_tokens = 10
		plot_tokens_per_attribute(attr_to_token, n_top_tokens, model_id, plots_dir)


	return (min_len,
			max_len,
			avg_len,
			n_utt)

def get_training_meters(model_id):
	model_dir = get_model_dir(model_id)

	losses_meters = get_pickle_file(model_dir, 'losses_meters.p')
	acc_meters = get_pickle_file(model_dir, 'accuracy_meters.p')
	entropy_meters = get_pickle_file(model_dir, 'entropy_meters.p')
	distinctness_meters = get_pickle_file(model_dir, 'distinctness_meters.p')
	rsa_sr_meters = get_pickle_file(model_dir, 'rsa_sr_meters.p')
	rsa_si_meters = get_pickle_file(model_dir, 'rsa_si_meters.p')
	rsa_ri_meters = get_pickle_file(model_dir, 'rsa_ri_meters.p')
	topological_sim_meters = get_pickle_file(model_dir, 'topological_sim_meters.p')

	return (losses_meters,
			acc_meters, 
			entropy_meters,
			distinctness_meters,
			rsa_sr_meters,
			rsa_si_meters,
			rsa_ri_meters,
			topological_sim_meters)

def plot_training_meters_curves(model_ids, analysis_id, plots_dir):
	output_dir = '{}/{}'.format(plots_dir, analysis_id)

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	for i,model_id in enumerate(model_ids):
		(losses_meters,
		acc_meters, 
		entropy_meters,
		distinctness_meters,
		rsa_sr_meters,
		rsa_si_meters,
		rsa_ri_meters,
		topological_sim_meters) = get_training_meters(model_id)

		assert (len(losses_meters) == len(acc_meters) and
				len(losses_meters) == len(entropy_meters) and
				len(losses_meters) == len(distinctness_meters) and
				len(losses_meters) == len(rsa_sr_meters) and
				len(losses_meters) == len(rsa_si_meters) and
				len(losses_meters) == len(topological_sim_meters))

		if i == 0:
			losses = np.array([m.avg for m in losses_meters])
			accuracies = np.array([m.avg for m in acc_meters])
			entropies = np.array([m.avg for m in entropy_meters])
			distincts = np.array([m.avg for m in distinctness_meters])
			rsa_srs = np.array([m.avg for m in rsa_sr_meters])
			rsa_sis = np.array([m.avg for m in rsa_si_meters])
			rsa_ris = np.array([m.avg for m in rsa_ri_meters])
			topo_similarities = np.array([m.avg for m in topological_sim_meters])
		else:
			losses_len = losses.shape[-1]

			if len(losses_meters) < losses_len:
				if losses.ndim == 1:
					losses = losses[:len(losses_meters)]
					accuracies = accuracies[:len(losses_meters)]
					entropies = entropies[:len(losses_meters)]
					distincts = distincts[:len(losses_meters)]
					rsa_srs = rsa_srs[:len(losses_meters)]
					rsa_sis = rsa_sis[:len(losses_meters)]
					rsa_ris = rsa_ris[:len(losses_meters)]
					topo_similarities = topo_similarities[:len(losses_meters)]
				else:
					assert losses.ndim == 2
					losses = losses[:, 0:len(losses_meters)]
					accuracies = accuracies[:, 0:len(losses_meters)]
					entropies = entropies[:, 0:len(losses_meters)]
					distincts = distincts[:, 0:len(losses_meters)]
					rsa_srs = rsa_srs[:, 0:len(losses_meters)]
					rsa_sis = rsa_sis[:, 0:len(losses_meters)]
					rsa_ris = rsa_ris[:, 0:len(losses_meters)]
					topo_similarities = topo_similarities[:, 0:len(losses_meters)]
			else:			
				losses_meters = losses_meters[:losses_len]
				acc_meters = acc_meters[:losses_len]
				entropy_meters = entropy_meters[:losses_len]
				distinctness_meters = distinctness_meters[:losses_len]
				rsa_sr_meters = rsa_sr_meters[:losses_len]
				rsa_si_meters = rsa_si_meters[:losses_len]
				rsa_ri_meters = rsa_ri_meters[:losses_len]
				topological_sim_meters = topological_sim_meters[:losses_len]

			losses = np.vstack((losses, np.array([m.avg for m in losses_meters])))
			accuracies = np.vstack((accuracies, np.array([m.avg for m in acc_meters])))
			entropies = np.vstack((entropies, np.array([m.avg for m in entropy_meters])))
			distincts = np.vstack((distincts, np.array([m.avg for m in distinctness_meters])))
			rsa_srs = np.vstack((rsa_srs, np.array([m.avg for m in rsa_sr_meters])))
			rsa_sis = np.vstack((rsa_sis, np.array([m.avg for m in rsa_si_meters])))
			rsa_ris = np.vstack((rsa_ris, np.array([m.avg for m in rsa_ri_meters])))
			topo_similarities = np.vstack((topo_similarities, np.array([m.avg for m in topological_sim_meters])))

	if losses.ndim > 1:
		assert (losses.shape[0] == len(model_ids) 
				and losses.shape == accuracies.shape 
				and losses.shape == entropies.shape
				and losses.shape == distincts.shape
				and losses.shape == rsa_srs.shape
				and losses.shape == rsa_sis.shape
				and losses.shape == rsa_ris.shape
				and losses.shape == topo_similarities.shape)

	if len(model_ids) > 1:
		losses_avg = np.mean(losses, axis=0)
		accuracies_avg = np.mean(accuracies, axis=0)
		entropies_avg = np.mean(entropies, axis=0)
		perplexities_avg = 2 ** entropies_avg
		distincts_avg = np.mean(distincts, axis=0)
		rsa_srs_avg = np.mean(rsa_srs, axis=0)
		rsa_sis_avg = np.mean(rsa_sis, axis=0)
		rsa_ris_avg = np.mean(rsa_ris, axis=0)
		topo_similarities_avg = np.mean(topo_similarities, axis=0)
	else:
		losses_avg = losses
		accuracies_avg = accuracies
		entropies_avg = entropies
		perplexities_avg = 2 ** entropies_avg
		distincts_avg = distincts
		rsa_srs_avg = rsa_srs
		rsa_sis_avg = rsa_sis
		rsa_ris_avg = rsa_ris
		topo_similarities_avg = topo_similarities


	iterations = range(len(losses_avg))

	# RSA plot
	plt.clf()
	plt.plot(iterations, rsa_srs_avg, color='blue')
	plt.plot(iterations, rsa_sis_avg, color='green')
	plt.plot(iterations, rsa_ris_avg, color='red')
	plt.legend(('Sender-Receiver', 'Sender-Input', 'Receiver-Input'))
	plt.xlabel('Epoch')
	plt.ylabel('RSA score')

	plt.savefig('{}/rsa_curves.png'.format(output_dir))

	# Acc, loss, message disntinctness plot
	plt.clf()
	plt.plot(iterations, accuracies_avg, color='blue')
	plt.plot(iterations, losses_avg, color='green')
	plt.plot(iterations, distincts_avg, color='red')
	plt.legend(('Accuracy', 'Loss', 'Message distinctness'))
	plt.xlabel('Epoch')

	plt.savefig('{}/acc_loss_curves.png'.format(output_dir))

	# Perplexity, topological similarities plot
	plt.clf()
	plt.plot(iterations, perplexities_avg, color='blue')
	plt.plot(iterations, topo_similarities_avg, color='green')
	plt.legend(('Perplexity', 'Topological similarity'))
	plt.xlabel('Epoch')

	plt.savefig('{}/perplexity_topo_sim_curves.png'.format(output_dir))

	print('Plots saved in {}'.format(output_dir))






# pickle.dump(losses_meters, open('{}/{}_{}_losses_meters.p'.format(current_model_dir, model_id, e), 'wb'))
# 	pickle.dump(eval_losses_meters, open('{}/{}_{}_eval_losses_meters.p'.format(current_model_dir, model_id, e), 'wb'))
# 	pickle.dump(accuracy_meters, open('{}/{}_{}_accuracy_meters.p'.format(current_model_dir, model_id, e), 'wb'))
# 	pickle.dump(eval_accuracy_meters, open('{}/{}_{}_eval_accuracy_meters.p'.format(current_model_dir, model_id, e), 'wb'))
# 	pickle.dump(entropy_meters, open('{}/{}_{}_entropy_meters.p'.format(current_model_dir, model_id, e), 'wb'))
# 	pickle.dump(eval_entropy_meters, open('{}/{}_{}_eval_entropy_meters.p'.format(current_model_dir, model_id, e), 'wb'))
# 	pickle.dump(distinctness_meters, open('{}/{}_{}_distinctness_meters.p'.format(current_model_dir, model_id, e), 'wb'))
# 	pickle.dump(eval_distinctness_meters, open('{}/{}_{}_eval_distinctness_meters.p'.format(current_model_dir, model_id, e), 'wb'))
# 	pickle.dump(rsa_sr_meters, open('{}/{}_{}_rsa_sr_meters.p'.format(current_model_dir, model_id, e), 'wb'))
# 	pickle.dump(eval_rsa_sr_meters, open('{}/{}_{}_eval_rsa_sr_meters.p'.format(current_model_dir, model_id, e), 'wb'))
# 	pickle.dump(rsa_si_meters, open('{}/{}_{}_rsa_si_meters.p'.format(current_model_dir, model_id, e), 'wb'))
# 	pickle.dump(eval_rsa_si_meters, open('{}/{}_{}_eval_rsa_si_meters.p'.format(current_model_dir, model_id, e), 'wb'))
# 	pickle.dump(rsa_ri_meters, open('{}/{}_{}_rsa_ri_meters.p'.format(current_model_dir, model_id, e), 'wb'))
# 	pickle.dump(eval_rsa_ri_meters, open('{}/{}_{}_eval_rsa_ri_meters.p'.format(current_model_dir, model_id, e), 'wb'))
# 	pickle.dump(topological_sim_meters, open('{}/{}_{}_topological_sim_meters.p'.format(current_model_dir, model_id, e), 'wb'))
# 	pickle.dump(eval_topological_sim_meters, open('{}/{}_{}_eval_topological_sim_meters.p'.format(current_model_dir, model_id, e), 'wb'))
