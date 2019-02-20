import pickle

VOCAB_SIZE = 100
START_TOKEN = '<S>'

idx_to_word = []
word_to_idx = {}

for i in range(VOCAB_SIZE - 1):
	word = str(i)
	idx_to_word.append(word)
	word_to_idx[word] = i

idx_to_word.append(START_TOKEN)
word_to_idx[START_TOKEN] = len(idx_to_word) - 1


with open('data/shapes/dict.pckl', 'wb') as f:
    pickle.dump({'word_to_idx': word_to_idx,
                  'idx_to_word': idx_to_word}, f)