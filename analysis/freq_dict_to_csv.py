import sys
import numpy as np
import pandas as pd
import os

from utils import get_pickle_file, get_model_dir

def get_readable_key(combination):
	color, shape = combination.split('_')
	color = int(color)
	shape = int(shape)

	if color == 0:
		color_str = 'red'
	elif color == 1:
		color_str = 'green'
	elif color == 2:
		color_str = 'blue'

	if shape == 0:
		shape_str = 'circle'
	elif shape == 1:
		shape_str = 'square'
	elif shape == 2:
		shape_str = 'triangle'

	return '{} {}'.format(color_str, shape_str)

csv_id = sys.argv[1]
model_ids = sys.argv[2:]

if not os.path.exists('tables/{}'.format(csv_id)):
	os.mkdir('tables/{}'.format(csv_id))

for model_id in model_ids:
	rsa_dicts = get_pickle_file(get_model_dir(model_id), 'test_frequency_based_rsa_topo.p')

	res_dict = {
		'Combination' : [],
		'Frequency' : [],
		'RSA Sender-Receiver' : [],
		'RSA Sender-Input' : [],
		'RSA Receiver-Input' : [],
		'Topological similarity': []
	}

	for key in rsa_dicts['shape_color_freq'].keys():
		res_dict['Combination'].append(get_readable_key(key))
		res_dict['Frequency'].append(rsa_dicts['shape_color_freq'][key])
		res_dict['RSA Sender-Receiver'].append(rsa_dicts['rsa_sr'][key])
		res_dict['RSA Sender-Input'].append(rsa_dicts['rsa_si'][key])
		res_dict['RSA Receiver-Input'].append(rsa_dicts['rsa_ri'][key])
		res_dict['Topological similarity'].append(rsa_dicts['topo'][key])

	# Dump sheet
	df = pd.DataFrame(res_dict)
	df.to_csv('tables/{}/{}_freq_based_rsa_topo.csv'.format(csv_id, model_id), index=None, header=True)


# 0501153314615371_25_10 0.1 1.0
# 0501132425775215_25_10 0.1 1.0
# 0501133707275044_25_10 0.1 1.0


# 0501134222757216_25_10 0 1
# 0501134635900332_25_10 0 1
# 0501135254792986_25_10 0 1


# 0501153314615371_25_10 0501132425775215_25_10 0501133707275044_25_10 0501134222757216_25_10 0501134635900332_25_10 0501135254792986_25_10

