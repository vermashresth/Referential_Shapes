from collections import Counter
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator


def load_dictionaries(vocab_size):
	with open("../data/shapes/dict_{}.pckl".format(vocab_size), "rb") as f:
	    d = pickle.load(f)
	    word_to_idx = d["word_to_idx"] # dictionary w->i
	    idx_to_word = d["idx_to_word"] # list of words
	    bound_idx = word_to_idx["<S>"] # last word in vocab

	return word_to_idx, idx_to_word, bound_idx

def get_message_length(message, padding_idx):	
	length = 0

	for i in range(len(message)):
		if message[i] == padding_idx:
			return length
		length += 1

	return length


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

	plt.savefig('{}/{}_token_dist.png'.format(plots_dir, model_id))


def plot_token_frequency(counter, n_utterances, percent, model_id, plots_dir):
	tokens = []
	occurrences = []

	for k, v in counter.most_common(int(n_utterances * percent)):
		tokens.append(k)
		occurrences.append(v)

	plt.clf()
	plt.title('Frequency for top {}% most used tokens'.format(int(percent * 100)))
	plt.ylabel('Number of occurrences')
	plt.xlabel('Token')
	plt.xticks(rotation=90)
	plt.bar(tokens, occurrences)

	plt.savefig('{}/{}_top_{}p_most_freq.png'.format(plots_dir, model_id, int(percent*100)))

# def plot_attributes_per_token(counter, n_utterances, percent, token_to_attr, model_id, plots_dir):
# 	tokens = []

# 	for token, _ in counter.most_common(int(n_utterances * percent)):
# 		tokens.append(token)

# 	plt.clf()
# 	plt.title('Attributes for top {}% most used tokens'.format(int(percent * 100)))
# 	plt.ylabel('Attributes')
# 	plt.xlabel('Token')
# 	plt.xticks(rotation=90)

# 	data_shape0 = [token_to_attr[t]['shape_0'] for t in tokens]
# 	data_shape1 = [token_to_attr[t]['shape_1'] for t in tokens]
# 	data_shape2 = [token_to_attr[t]['shape_2'] for t in tokens]
# 	data_color0 = [token_to_attr[t]['color_0'] for t in tokens]
# 	data_color1 = [token_to_attr[t]['color_1'] for t in tokens]
# 	data_color2 = [token_to_attr[t]['color_2'] for t in tokens]
# 	data_size0 = [token_to_attr[t]['size_0'] for t in tokens]
# 	data_size1 = [token_to_attr[t]['size_1'] for t in tokens]

# 	pshape0 = plt.bar(tokens, data_shape0)
# 	pshape1 = plt.bar(tokens, data_shape1, bottom=data_shape0)
# 	pshape2 = plt.bar(tokens, data_shape2, bottom=data_shape1)
# 	pcolor0 = plt.bar(tokens, data_color0, bottom=data_shape2, color='red')
# 	pcolor1 = plt.bar(tokens, data_color1, bottom=data_color0, color='green')
# 	pcolor2 = plt.bar(tokens, data_color2, bottom=data_color1, color='blue')
# 	psize0 = plt.bar(tokens, data_size0, bottom=data_color2)
# 	psize1 = plt.bar(tokens, data_size1, bottom=data_size0)

# 	plt.legend(
# 		(pshape0[0], pshape1[0], pshape2[0], pcolor0[0], pcolor1[0], pcolor2[0], psize0[0], psize1[0]), 
# 		('circle', 'square', 'triangle', 'red', 'green', 'blue', 'small', 'big'))

# 	plt.savefig('{}/{}_attributes_per_top_{}p_tokens.png'.format(plots_dir, model_id, int(percent*100)))

def get_bar_color(name):
	if name == 'red':
		return 'r'
	if name == 'green':
		return 'g'
	if name == 'blue':
		return 'b'
	if name == 'circle':
		return 'c'
	if name == 'square':
		return 'y'
	if name == 'triangle':
		return 'm'
	if name == 'small':
		return 'y'
	if name == 'big':
		return 'k'
	assert False, 'Should not be here'

def plot_tokens_per_attribute(attr_to_token, n_top_tokens, model_id, plots_dir):
	plt.clf()
	plt.figure(figsize=(12,9)) # Originally 8x6 (inches)
	plt.title('Top tokens per attribute')

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

	for attr_id, attr in attrs_strings.items():
		c = attr_to_token[attr_id]
		tokens = []
		occurrences = []
		for token, v in c.most_common(n_top_tokens):
			tokens.append(token)
			occurrences.append(v)

		plt.subplot(n_rows, n_columns, pos, title=attr)
		pos += 1
		plt.bar(tokens, occurrences, color=get_bar_color(attr))
		plt.ylabel('Occurrences')
		# plt.xlabel('Tokens')
		plt.xticks(rotation=90)

	plt.subplots_adjust(hspace=0.5)

	plt.savefig('{}/{}_top_tokens_per_attr.png'.format(plots_dir, model_id))


def get_feature_ids(message_metadata):
	ids = []
	for k, v_list in message_metadata.items():
		flat_list = [item for sublist in v_list for item in sublist]
		for v in flat_list:
			if not v == None:
				ids.append('{}_{}'.format(k[:-1], v))

	return ids

def get_attributes_dicts(messages, metadata, idx_to_word):
	token_to_attr = {}
	attr_to_token = {}

	for i, m in enumerate(messages):
		feature_keys = get_feature_ids(metadata[i])

		for token_idx in m:
			token = idx_to_word[token_idx]
			if not token in token_to_attr:
				token_to_attr[token] = Counter()
			
			for feature_key in feature_keys:
				token_to_attr[token][feature_key] += 1

				if not feature_key in attr_to_token:
					attr_to_token[feature_key] = Counter()

				attr_to_token[feature_key][token] += 1

	return token_to_attr, attr_to_token

def plot_acc_per_setting(stats, dir, id):
	plt.clf()
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	
	ax.scatter([int(x) for x in stats['|V|']], 
				[int(x) for x in stats['L']], 
				stats['Test acc'])

	ax.set_xlabel('|V|')
	ax.set_ylabel('L')
	ax.set_zlabel('Accuracy')

	ax.yaxis.set_major_locator(MaxNLocator(integer=True))
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))

	plt.savefig('{}/acc_per_setting_{}.png'.format(dir, id))



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




# attr_to_token = {
# 	'shape_0': Counter(),
# 	'shape_1': Counter(),
# 	'shape_2': Counter(),
# 	'color_0': Counter(),
# 	'color_1': Counter(),
# 	'color_2': Counter(),
# 	'size_0': Counter(),
# 	'size_1': Counter()
# }

# for i, k in enumerate(attr_to_token):
# 	for j in range(10):
# 		attr_to_token[k]['{}'.format(j)] += np.random.randint(1, 50)



# n_top_tokens = 10
# plot_tokens_per_attribute(attr_to_token, n_top_tokens, 'dummy_id', 'plots')