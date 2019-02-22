from collections import Counter
import matplotlib.pyplot as plt
import pickle

def load_dictionaries(vocab_size):
	with open("../data/shapes/dict_{}.pckl".format(vocab_size), "rb") as f:
	    d = pickle.load(f)
	    word_to_idx = d["word_to_idx"] # dictionary w->i
	    idx_to_word = d["idx_to_word"] # list of words
	    bound_idx = word_to_idx["<S>"] # last word in vocab

	return word_to_idx, idx_to_word, bound_idx

def get_message_length(message, padding_idx):
	assert message[0] == padding_idx, 'First token should always be <S>!'
	
	start_pad_pos = 0

	for i in reversed(range(len(message))):
		if message[i] == padding_idx:
			start_pad_pos = i
		else:
			break

	# we ignore the first token for statistical purposes
	if start_pad_pos == 0:
		return len(message) - 1
	else:
		return start_pad_pos - 1


def compute_stats(messages, padding_idx, idx_to_word):
	max_len = -1
	min_len = len(messages[0])
	avg_len = 0

	c = Counter()

	for i,m in enumerate(messages):		
		length = get_message_length(m, padding_idx)
		if length > max_len:
			max_len = length
		if length < min_len:
			min_len = length
		avg_len = avg_len*((i)/(i+1)) + length*(1/(i+1))

		for token_idx in m:
			c[idx_to_word[token_idx]] += 1

	return min_len, max_len, avg_len, c


def plot_token_distribution(counter, model_id, plots_dir):
	n_tokens_limit = 500
	tokens = []
	occurrences = []

	if len(counter) > n_tokens_limit:
		items = counter.most_common(n_tokens_limit)
	else:
		items = counter.most_common()
	
	for k,v in items:
		tokens.append(k)
		occurrences.append(v)

	# Frequency rank
	plt.clf()
	plt.title('Tokens distribution{}'.format(' (partial)' if len(counter) > n_tokens_limit else ''))
	plt.ylabel('Number of occurrences')
	plt.xlabel('Token')
	plt.xticks(rotation=90)
	plt.plot(tokens, occurrences)

	plt.savefig('{}/token_dist_{}.png'.format(plots_dir, model_id))


def plot_token_frequency(counter, n_utterances, percent, model_id, plots_dir):
	tokens = []
	occurrences = []

	for k, v in counter.most_common(int(n_utterances * percent)):
		tokens.append(k)
		occurrences.append(v)

	plt.clf()
	plt.title('Frequency for top {}% most used tokens'.format(percent * 100))
	plt.ylabel('Number of occurrences')
	plt.xlabel('Token')
	plt.xticks(rotation=90)
	plt.bar(tokens, occurrences)

	plt.savefig('{}/top_{}percent_most_freq_{}.png'.format(plots_dir, int(percent*100), model_id))

def plot_attributes_per_token(counter, n_utterances, percent, token_to_attr, model_id, plots_dir):
	tokens = []

	for token, _ in counter.most_common(int(n_utterances * percent)):
		if token == '<S>':
			continue
		tokens.append(token)

	plt.clf()
	plt.title('Attributes for top {}% most used tokens'.format(int(percent * 100)))
	plt.ylabel('Attributes')
	plt.xlabel('Token')
	plt.xticks(rotation=90)

	data_shape0 = [token_to_attr[t]['shape_0'] for t in tokens]
	data_shape1 = [token_to_attr[t]['shape_1'] for t in tokens]
	data_shape2 = [token_to_attr[t]['shape_2'] for t in tokens]
	data_color0 = [token_to_attr[t]['color_0'] for t in tokens]
	data_color1 = [token_to_attr[t]['color_1'] for t in tokens]
	data_color2 = [token_to_attr[t]['color_2'] for t in tokens]
	data_size0 = [token_to_attr[t]['size_0'] for t in tokens]
	data_size1 = [token_to_attr[t]['size_1'] for t in tokens]

	pshape0 = plt.bar(tokens, data_shape0)
	pshape1 = plt.bar(tokens, data_shape1, bottom=data_shape0)
	pshape2 = plt.bar(tokens, data_shape2, bottom=data_shape1)
	pcolor0 = plt.bar(tokens, data_color0, bottom=data_shape2)
	pcolor1 = plt.bar(tokens, data_color1, bottom=data_color0)
	pcolor2 = plt.bar(tokens, data_color2, bottom=data_color1)
	psize0 = plt.bar(tokens, data_size0, bottom=data_color2)
	psize1 = plt.bar(tokens, data_size1, bottom=data_size0)

	plt.legend(
		(pshape0[0], pshape1[0], pshape2[0], pcolor0[0], pcolor1[0], pcolor2[0], psize0[0], psize1[0]), 
		('circle', 'square', 'triangle', 'red', 'green', 'blue', 'small', 'big'))

	plt.savefig('{}/attributes_per_top_{}p_tokens_{}.png'.format(plots_dir, int(percent*100), model_id))


def plot_tokens_per_attribute(attr_to_token, n_top_tokens, model_id, plots_dir):
	plt.clf()
	plt.figure(figsize=(12,9)) # Originally 8x6 (inches)
	plt.title('Top {} tokens per attribute'.format(n_top_tokens))

	# 8 attributes
	n_rows = 4
	n_columns = 2
	pos = 1

	attrs_strings = {'shape_0': 'circle',
					 'shape_1': 'square',
					 'shape_2': 'triangle',
					 'color_0': 'red',
					 'color_1': 'green',
					 'color_2': 'blue',
					 'size_0': 'small',
					 'size_1': 'big'}

	for attr_id, title in attrs_strings.items():
		c = attr_to_token[attr_id]
		tokens = []
		occurrences = []
		for token, v in c.most_common(n_top_tokens):
			if token == '<S>':
				continue

			tokens.append(token)
			occurrences.append(v)

		plt.subplot(n_rows, n_columns, pos, title=title)
		pos += 1
		plt.bar(tokens, occurrences)
		plt.ylabel('Occurrences')
		plt.xlabel('Tokens')
		plt.xticks(rotation=90)

	plt.savefig('{}/top_{}_tokens_per_attribute.png'.format(plots_dir, n_top_tokens))


def get_feature_ids(message_metadata):
	ids = []
	for k, v_list in message_metadata.items():
		flat_list = [item for sublist in v_list for item in sublist]
		for v in flat_list:
			if not v == None:
				ids.append('{}_{}'.format(k[:-1], v))

	return ids

def get_attributes_dicts(messages, padding_idx, metadata, idx_to_word):
	token_to_attr = {}
	attr_to_token = {}

	for i, m in enumerate(messages):
		feature_keys = get_feature_ids(metadata[i])

		for token_idx in m:
			if token_idx == padding_idx:
				continue

			token = idx_to_word[token_idx]
			if not token in token_to_attr:
				token_to_attr[token] = Counter()
			
			for feature_key in feature_keys:
				token_to_attr[token][feature_key] += 1

				if not feature_key in attr_to_token:
					attr_to_token[feature_key] = Counter()

				attr_to_token[feature_key][token] += 1

	return token_to_attr, attr_to_token

class AverageMeter:
    def __init__(self):
        self.value = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
