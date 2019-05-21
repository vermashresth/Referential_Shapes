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
	lang_entropy_meter = get_pickle_file(model_dir, 'test_language_entropy_meter.p')

	return (acc_meter.avg,
			entropy_meter.avg,
			distinctness_meter.avg,
			rsa_sr_meter.avg,
			rsa_si_meter.avg,
			rsa_ri_meter.avg,
			topological_sim_meter.avg,
			lang_entropy_meter.avg)

def get_test_messages_stats(model_id, vocab_size, data_folder, plots_dir, should_plot=False):
	_, idx_to_word, padding_idx = load_dictionaries('shapes' if '_' in data_folder else 'mscoco', vocab_size)

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

def get_training_meters(model_id, debugging=False):
	model_dir = get_model_dir(model_id)

	if not debugging:
		# rsa_srs_list = get_pickle_file(model_dir, 'rsa_srs_list.p')
		# acc_meters = get_pickle_file(model_dir, 'accuracy_meters.p')
		# entropy_meters = get_pickle_file(model_dir, 'entropy_meters.p')
		# distinctness_meters = get_pickle_file(model_dir, 'distinctness_meters.p')
		rsa_sr_meters = get_pickle_file(model_dir, 'rsa_sr_meters.p')
		rsa_si_meters = get_pickle_file(model_dir, 'rsa_si_meters.p')
		rsa_ri_meters = get_pickle_file(model_dir, 'rsa_ri_meters.p')
		topological_sim_meters = get_pickle_file(model_dir, 'topological_sim_meters.p')
		# language_entropy_meters = get_pickle_file(model_dir, 'language_entropy_meters.p')
	else:
		# rsa_srs_list = [AverageMeter() for _ in range(40)]
		# acc_meters = [AverageMeter() for _ in range(40)]
		# entropy_meters = [AverageMeter() for _ in range(40)]
		# distinctness_meters = [AverageMeter() for _ in range(40)]
		rsa_sr_meters = [AverageMeter() for _ in range(40)]
		rsa_si_meters = [AverageMeter() for _ in range(40)]
		rsa_ri_meters = [AverageMeter() for _ in range(40)]
		topological_sim_meters = [AverageMeter() for _ in range(40)]
		# language_entropy_meters = [AverageMeter() for _ in range(40)]

		for meter_list in [rsa_srs_list, acc_meters, distinctness_meters, rsa_sr_meters, rsa_si_meters, rsa_ri_meters, topological_sim_meters]:
			for m in meter_list:
				m.update(np.random.random())

		# for meter_list in [entropy_meters, language_entropy_meters]:
		# 	for m in meter_list:
		# 		m.update(np.random.random() + (1 if np.random.random() < 0.5 else 2))

	return (#rsa_srs_list,
			#acc_meters, 
			#entropy_meters,
			#distinctness_meters,
			rsa_sr_meters,
			rsa_si_meters,
			rsa_ri_meters,
			topological_sim_meters,
			#language_entropy_meters
			)

def get_training_values(model_id, per_epoch, debugging=False):
	model_dir = get_model_dir(model_id)

	rsa_sr_meters = get_pickle_file(model_dir, 'rsa_sr_meters.p')
	rsa_si_meters = get_pickle_file(model_dir, 'rsa_si_meters.p')
	rsa_ri_meters = get_pickle_file(model_dir, 'rsa_ri_meters.p')
	topological_sim_meters = get_pickle_file(model_dir, 'topological_sim_meters.p')

	if per_epoch:
		return ([m.avg for m in rsa_sr_meters],
				[m.avg for m in rsa_si_meters],
				[m.avg for m in rsa_ri_meters],
				[m.avg for m in topological_sim_meters])
	else:
		all_datapoints = []
		for meters_list in [rsa_sr_meters, rsa_si_meters, rsa_ri_meters, topological_sim_meters]:
			datapoints = []
			for m in meters_list:
				datapoints.extend(m.all_values)
			all_datapoints.append(datapoints)

		return (all_datapoints[0],
				all_datapoints[1],
				all_datapoints[2],
				all_datapoints[3])

def plot_rsa_topo_curves(model_dict, analysis_id, plots_dir, debugging=False):
	output_dir = '{}/{}'.format(plots_dir, analysis_id)

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	metrics_per_dataset = {k:{'rsa_sr':None, 'rsa_si':None, 'rsa_ri':None, 'topo_sim':None} for k in set(model_dict['dataset'])}

	for i, model_id in enumerate(model_dict['id']):
		dataset_id = model_dict['dataset'][i]

		(rsa_srs_list,
		rsa_sis_list,
		rsa_ris_list,
		topological_sims_list) = get_training_values(model_id, per_epoch=True, debugging=debugging)

		for l in [rsa_sis_list, rsa_ris_list, topological_sims_list]:
			assert len(rsa_srs_list) == len(l)

		if metrics_per_dataset[dataset_id]['rsa_sr'] is None:
			metrics_per_dataset[dataset_id]['rsa_sr'] = np.array(rsa_srs_list)
			metrics_per_dataset[dataset_id]['rsa_si'] = np.array(rsa_sis_list)
			metrics_per_dataset[dataset_id]['rsa_ri'] = np.array(rsa_ris_list)
			metrics_per_dataset[dataset_id]['topo_sim'] = np.array(topological_sims_list)
		else:
			length = metrics_per_dataset[dataset_id]['rsa_sr'].shape[-1]

			if len(rsa_srs_list) < length:
				if metrics_per_dataset[dataset_id]['rsa_sr'].ndim == 1:
					metrics_per_dataset[dataset_id]['rsa_sr'] = metrics_per_dataset[dataset_id]['rsa_sr'][:len(rsa_srs_list)]
					metrics_per_dataset[dataset_id]['rsa_si'] = metrics_per_dataset[dataset_id]['rsa_si'][:len(rsa_srs_list)]
					metrics_per_dataset[dataset_id]['rsa_ri'] = metrics_per_dataset[dataset_id]['rsa_ri'][:len(rsa_srs_list)]
					metrics_per_dataset[dataset_id]['topo_sim'] = metrics_per_dataset[dataset_id]['topo_sim'][:len(rsa_srs_list)]
				else:
					assert metrics_per_dataset[dataset_id]['rsa_sr'].ndim == 2
					metrics_per_dataset[dataset_id]['rsa_sr'] = metrics_per_dataset[dataset_id]['rsa_sr'][:, 0:len(rsa_srs_list)]
					metrics_per_dataset[dataset_id]['rsa_si'] = metrics_per_dataset[dataset_id]['rsa_si'][:, 0:len(rsa_srs_list)]
					metrics_per_dataset[dataset_id]['rsa_ri'] = metrics_per_dataset[dataset_id]['rsa_ri'][:, 0:len(rsa_srs_list)]
					metrics_per_dataset[dataset_id]['topo_sim'] = metrics_per_dataset[dataset_id]['topo_sim'][:, 0:len(rsa_srs_list)]
			else:
				rsa_srs_list = rsa_srs_list[:length]
				rsa_sis_list = rsa_sis_list[:length]
				rsa_ris_list = rsa_ris_list[:length]
				topological_sims_list = topological_sims_list[:length]

			metrics_per_dataset[dataset_id]['rsa_sr'] = np.vstack((metrics_per_dataset[dataset_id]['rsa_sr'], np.array(rsa_srs_list)))
			metrics_per_dataset[dataset_id]['rsa_si'] = np.vstack((metrics_per_dataset[dataset_id]['rsa_si'], np.array(rsa_sis_list)))
			metrics_per_dataset[dataset_id]['rsa_ri'] = np.vstack((metrics_per_dataset[dataset_id]['rsa_ri'], np.array(rsa_ris_list)))
			metrics_per_dataset[dataset_id]['topo_sim'] = np.vstack((metrics_per_dataset[dataset_id]['topo_sim'], np.array(topological_sims_list)))

	metrics_avg_per_dataset = {}

	for dataset, dictionary in metrics_per_dataset.items():
		n_models = model_dict['dataset'].count(dataset)

		if dictionary['rsa_sr'].ndim > 1:
			assert (dictionary['rsa_sr'].shape[0] == n_models
					and dictionary['rsa_sr'].shape == dictionary['rsa_si'].shape
					and dictionary['rsa_sr'].shape == dictionary['rsa_ri'].shape
					and dictionary['rsa_sr'].shape == dictionary['topo_sim'].shape)

		if n_models > 1:
			rsa_srs_avg = np.mean(dictionary['rsa_sr'], axis=0)
			rsa_sis_avg = np.mean(dictionary['rsa_si'], axis=0)
			rsa_ris_avg = np.mean(dictionary['rsa_ri'], axis=0)
			topo_similarities_avg = np.mean(dictionary['topo_sim'], axis=0)
		else:
			rsa_srs_avg = dictionary['rsa_sr']
			rsa_sis_avg = dictionary['rsa_si']
			rsa_ris_avg = dictionary['rsa_ri']
			topo_similarities_avg = dictionary['topo_sim']


		metrics_avg_per_dataset[dataset] = {
			'rsa_sr' : rsa_srs_avg,
			'rsa_si' : rsa_sis_avg,
			'rsa_ri' : rsa_ris_avg,
			'topo_sim' : topo_similarities_avg
		}

	
	# print("Planning to plot")
	# print()
	# print(metrics_avg_per_dataset)

	colors = {}
	colors['blue_dark'] = (7, 13, 79)
	colors['blue_medium'] = (6, 29, 229)
	colors['blue_light'] = (158, 200, 239)
	colors['green'] = (44, 132, 33)
	colors['green_dark'] = (27, 89, 17)
	colors['green_medium'] = (124, 219, 109)
	colors['orange'] = (255, 106, 0)
	colors['red_light'] = (219, 87, 87)
	colors['red_dark'] = (219, 8, 8)
	colors['purple'] = (139, 116, 173)

	for k in colors.keys():
		colors[k] = (colors[k][0] / 255, colors[k][1] / 255, colors[k][2] / 255)

	if debugging:
		plt.close('all')

	font_dict = {'family': 'serif'}

	plt.clf()
	fig, ax = plt.subplots(figsize=(16,4))

	ax.plot(metrics_avg_per_dataset['balanced_3_3']['rsa_si'], color=colors['blue_light'], label='Baseline Sender-Input')
	ax.plot(metrics_avg_per_dataset['balanced_3_3']['rsa_ri'], color=colors['blue_dark'], label='Baseline Receiver-Input')
	ax.plot(metrics_avg_per_dataset['different_targets_3_3']['rsa_si'], color=colors['green_medium'], label='Diff targets Sender-Input')
	ax.plot(metrics_avg_per_dataset['different_targets_3_3']['rsa_ri'], color=colors['green_dark'], label='Diff targets Receiver-Input')
	ax.plot(metrics_avg_per_dataset['uneven_3_3']['rsa_si'], color=colors['red_light'], label='Skewed Sender-Input')
	ax.plot(metrics_avg_per_dataset['uneven_3_3']['rsa_ri'], color=colors['red_dark'], label='Skewed Receiver-Input')

	
	plt.xlabel('Epoch') # 'Iteration'
	plt.grid(True)
	plt.rc('font', family='serif')

	plt.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., prop=font_dict)

	plt.savefig('{}/rsa_training_curves.png'.format(output_dir), bbox_inches='tight')

	if debugging:
		plt.show()

	plt.clf()
	fig, ax = plt.subplots(figsize=(16,4))

	ax.plot(metrics_avg_per_dataset['balanced_3_3']['topo_sim'], color=colors['purple'], label='Baseline Topographic similarity')
	ax.plot(metrics_avg_per_dataset['different_targets_3_3']['topo_sim'], color=colors['orange'], label='Diff targets Topographic similarity')
	ax.plot(metrics_avg_per_dataset['uneven_3_3']['topo_sim'], color=colors['blue_light'], label='Skewed Topographic similarity')

	plt.xlabel('Epoch')
	plt.grid(True)
	plt.rc('font', family='serif')

	plt.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., prop=font_dict)

	plt.savefig('{}/topographic_sim_training_curves.png'.format(output_dir), bbox_inches='tight')

	if debugging:
		plt.show()

	print('Plots saved in {}'.format(output_dir))















# def plot_training_meters_curves(model_ids, analysis_id, plots_dir, debugging=false):
# 	output_dir = '{}/{}'.format(plots_dir, analysis_id)

# 	if not os.path.exists(output_dir):
# 		os.mkdir(output_dir)

# 	for i, model_id in enumerate(model_ids):
# 		(rsa_srs_list,
# 		rsa_sis_list,
# 		rsa_ris_list,
# 		topological_sims_list) = get_training_values(model_id, per_epoch=true, debugging=debugging)

# 		for l in [rsa_sis_list, rsa_ris_list, topological_sims_list]:
# 			assert len(rsa_srs_list) == len(l)

# 		if i == 0:
# 			rsa_srs = np.array(rsa_srs_list)
# 			rsa_sis = np.array(rsa_sis_list)
# 			rsa_ris = np.array(rsa_ris_list)
# 			topo_similarities = np.array(topological_sims_list)
# 		else:
# 			length = rsa_srs.shape[-1]

# 			if len(rsa_srs_list) < length:
# 				if rsa_srs.ndim == 1:
# 					rsa_srs = rsa_srs[:len(rsa_srs_list)]
# 					rsa_sis = rsa_sis[:len(rsa_srs_list)]
# 					rsa_ris = rsa_ris[:len(rsa_srs_list)]
# 					topo_similarities = topo_similarities[:len(rsa_srs_list)]
# 				else:
# 					assert rsa_srs.ndim == 2
# 					rsa_srs = rsa_srs[:, 0:len(rsa_srs_list)]
# 					rsa_sis = rsa_sis[:, 0:len(rsa_srs_list)]
# 					rsa_ris = rsa_ris[:, 0:len(rsa_srs_list)]
# 					topo_similarities = topo_similarities[:, 0:len(rsa_srs_list)]
# 			else:
# 				rsa_srs_list = rsa_srs_list[:length]
# 				rsa_sis_list = rsa_sis_list[:length]
# 				rsa_ris_list = rsa_ris_list[:length]
# 				topological_sims_list = topological_sims_list[:length]

# 			rsa_srs = np.vstack((rsa_srs, np.array(rsa_srs_list)))
# 			rsa_sis = np.vstack((rsa_sis, np.array(rsa_sis_list)))
# 			rsa_ris = np.vstack((rsa_ris, np.array(rsa_ris_list)))
# 			topo_similarities = np.vstack((topo_similarities, np.array(topological_sims_list)))

# 	if rsa_srs.ndim > 1:
# 		assert (rsa_srs.shape[0] == len(model_ids)
# 				and rsa_srs.shape == rsa_sis.shape
# 				and rsa_srs.shape == rsa_ris.shape
# 				and rsa_srs.shape == topo_similarities.shape)

# 	if len(model_ids) > 1:
# 		rsa_srs_avg = np.mean(rsa_srs, axis=0)
# 		rsa_sis_avg = np.mean(rsa_sis, axis=0)
# 		rsa_ris_avg = np.mean(rsa_ris, axis=0)
# 		topo_similarities_avg = np.mean(topo_similarities, axis=0)
# 	else:
# 		rsa_srs_avg = rsa_srs
# 		rsa_sis_avg = rsa_sis
# 		rsa_ris_avg = rsa_ris
# 		topo_similarities_avg = topo_similarities

# 	# perplexities_avg /= 20 # obverters?
# 	# lang_entropies_avg /= 20


# 	iterations = range(len(rsa_srs_avg))

# 	colors = {}
# 	colors['blue_dark'] = (7, 13, 79)
# 	colors['blue_medium'] = (6, 29, 229)
# 	colors['blue_light'] = (158, 200, 239)
# 	colors['green'] = (44, 132, 33)
# 	colors['orange'] = (255, 106, 0)
# 	colors['red'] = (253, 6, 6)
# 	colors['purple'] = (139, 116, 173)

# 	for k in colors.keys():
# 		colors[k] = (colors[k][0] / 255, colors[k][1] / 255, colors[k][2] / 255)

	
# 	if debugging:
# 		plt.close('all')


# 	font_dict = {'family': 'serif'}

# 	plt.clf()
# 	fig, ax = plt.subplots(figsize=(16,4))

# 	ax.plot(iterations, rsa_srs_avg, color=colors['blue_medium'], label='rsa sender-receiver')
# 	ax.plot(iterations, rsa_sis_avg, color=colors['green'], label='rsa sender-input')
# 	ax.plot(iterations, rsa_ris_avg, color=colors['red'], label='rsa receiver-input')
	
# 	plt.xlabel('epoch') # 'iteration'
# 	plt.grid(true)
# 	plt.rc('font', family='serif')

# 	plt.legend(frameon=false, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., prop=font_dict)

# 	plt.savefig('{}/rsa_training_curves.png'.format(output_dir), bbox_inches='tight')

# 	if debugging:
# 		plt.show()

# 	plt.clf()
# 	fig, ax = plt.subplots(figsize=(16,4))

# 	ax.plot(iterations, topo_similarities_avg, color=colors['purple'], label='topographic similarity')
# 	# ax.plot(iterations, distincts_avg, color=colors['purple'], label='message distinctness')
# 	# ax.plot(iterations, perplexities_avg, color=colors['red'], label='perplexity per symbol')
# 	# ax.plot(iterations, lang_entropies_avg, color=colors['orange'], label='language entropy')

# 	plt.xlabel('epoch')
# 	plt.grid(true)
# 	plt.rc('font', family='serif')

# 	plt.legend(frameon=false, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., prop=font_dict)

# 	plt.savefig('{}/topographic_sim_training_curves.png'.format(output_dir), bbox_inches='tight')

# 	if debugging:
# 		plt.show()

# 	print('plots saved in {}'.format(output_dir))
# 	