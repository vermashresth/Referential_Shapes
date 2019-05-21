import sys
import numpy as np

from utils import get_pickle_file, get_model_dir

n_samples = int(sys.argv[1])
model_ids = sys.argv[2:]

for model_id in model_ids:
	m = get_pickle_file(get_model_dir(model_id), 'test_messages_w.p')

	idxs = range(n_samples)

	print('Model id: {}'.format(model_id))
	for idx in idxs:
		print(m[idx])



