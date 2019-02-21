from collections import Counter
import matplotlib.pyplot as plt


def get_message_length(message, padding_token):
	assert message[0] == padding_token
	
	pad_idx = 0

	for i in reversed(range(len(message))):
		assert False, 'Need to check if message[i] I need to do .item() or padding_token is what???'
		if message[i] == padding_token:
			pad_idx = i
		else:
			break

	# we ignore the first token for statistical purposes
	if pad_idx == 0:
		return len(message) - 1
	else:
		return pad_idx - 1


def compute_stats(messages, padding_token):
	max_len = -1
	min_len = len(messages[0])
	avg_len = 0

	c = Counter()

	for i,m in enumerate(messages):		
		length = get_message_length(m, padding_token)
		if length > max_len:
			max_len = length
		if length < min_len:
			min_len = length
		avg_len = avg_len*((i)/(i+1)) + length*(1/(i+1))

		for token in m:
			c[token.item()] += 1

	return min_len, max_len, avg_len, c


def plot_token_distribution(counter, model_id):
	n_tokens_limit = 500
	tokens = []
	occurrences = []

	if len(counter) > n_tokens_limit:
		for k,v in counter.most_common(n_tokens_limit):
			tokens.append(str(k))
			occurrences.append(v)
	else:
		for k,v in counter.most_common():
			tokens.append(str(k))
			occurrences.append(v)

	# Frequency rank
	plt.title('Tokens distribution{}'.format(' (partial)' if len(counter) > n_tokens_limit else ''))
	plt.ylabel('Number of occurrences')
	plt.xlabel('Token')
	plt.plot(tokens, occurrences)

	plt.savefig('token_dist_{}.png'.format(model_id))

def get_feature_ids(message_metadata):
	ids = []
	for k, v_list in message_metadata.items():
		flat_list = [item for sublist in v_list for item in sublist]
		for v in flat_list:
			if not v == None:
				ids.append('{}_{}'.format(k[:-1], v))

	return ids

def get_attributes(messages, padding_token, metadata):
	dict = {}

	for i, m in enumerate(messages):
		for t in m:
			if t == padding_token: ## Is this correct? or need .item() ?
				continue

			token = t.item()
			if not token in dict:
				dict[token] = Counter()
			feature_keys = get_feature_ids(metadata[i])
			for feature_key in feature_keys:
				dict[token][feature_key] += 1

	return dict

