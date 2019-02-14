import pickle
import torch
import os

from model import Model
from run import train_one_epoch, evaluate
from dataloader import load_dictionaries, load_shapes_data, load_data

use_gpu = torch.cuda.is_available()

EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 128 if use_gpu else 4
MAX_SENTENCE_LENGTH = 13 if use_gpu else 5
START_TOKEN = '<S>'
K = 3

seed = 42
torch.manual_seed(seed)
if use_gpu:
	torch.cuda.manual_seed(seed)

best_model_name = 'dumps/02_09_19_44/02_09_19_44_106_model'

# Load vocab
word_to_idx, idx_to_word, bound_idx = load_dictionaries()
vocab_size = len(word_to_idx) # 10000

# Load data
n_image_features, _, _, test_data = load_data(BATCH_SIZE, K)


# Settings
dumps_dir = './dumps'
if not os.path.exists(dumps_dir):
	os.mkdir(dumps_dir)


last_backslash = best_model_name.rfind('/')
last_underscore = best_model_name.rfind('_')
second_last_underscore = best_model_name[:last_underscore].rfind('_')
model_id = best_model_name[last_backslash+1:second_last_underscore]
best_epoch = int(best_model_name[second_last_underscore+1:last_underscore])

current_model_dir = '{}/{}'.format(dumps_dir, model_id)

if not os.path.exists(current_model_dir):
	os.mkdir(current_model_dir)


# Load model
best_model = Model(n_image_features, vocab_size,
	EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, use_gpu)
state = torch.load(best_model_name, map_location= lambda storage, location: storage)
best_model.load_state_dict(state)

if use_gpu:
	best_model = best_model.cuda()

_, test_acc_meter, test_messages = evaluate(best_model, test_data, word_to_idx, START_TOKEN, MAX_SENTENCE_LENGTH)

print('Test accuracy: {}'.format(test_acc_meter.avg))

pickle.dump(test_acc_meter, open('{}/{}_{}_test_accuracy_meter2.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
pickle.dump(test_messages, open('{}/{}_{}_test_messages.p'.format(current_model_dir, model_id, best_epoch), 'wb'))


# test_messages = pickle.load(open('{}/{}_{}_test_messages.p'.format(current_model_dir, model_id, best_epoch), 'rb'))
#print(len(test_messages), 'test messages')

# Decode test messages
words = []

for m in test_messages:
	words.append([idx_to_word[t] for t in m])

pickle.dump(words, open('{}/{}_{}_test_messages_w.p'.format(current_model_dir, model_id, best_epoch), 'wb'))