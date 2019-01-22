import json
import pickle as cPickle
import numpy as np
from collections import Counter


rnd = np.random.RandomState(42)
features = np.load('data/mscoco/raw/mscoco_train.npy')
test_features = np.load('data/mscoco/raw/mscoco_val.npy')
with open('data/mscoco/raw/mscoco.json') as f:
    captions = json.load(f)
test_captions = captions['val']

indxs = list(range(features.shape[0]))
rnd.shuffle(indxs)

train_indxs = indxs[:int(len(indxs) * 0.9)]
train_captions = [captions['train'][idx] for idx in train_indxs]
train_features = features[train_indxs]

valid_indxs = indxs[int(len(indxs) * 0.9):]
valid_captions = [captions['train'][idx] for idx in valid_indxs]
valid_features = features[valid_indxs]


c = Counter()
for e in train_captions:
    for sentence in e[1]:
        for word in sentence:
            c[word.lower()] += 1
idx_to_word = [e[0] for e in c.most_common(9998)]
word_to_idx = dict((word, i) for i, word in enumerate(idx_to_word))
word_to_idx['<UNK>'] = len(idx_to_word)
idx_to_word.append('<UNK>')
word_to_idx['<S>'] = len(idx_to_word)
idx_to_word.append('<S>')


unk_idx = word_to_idx['<UNK>']
s_idx = word_to_idx['<S>']
for cap in [train_captions, valid_captions, test_captions]:
    for e in cap:
        for sentence in e[1]:
            for i in range(len(sentence)):
                word = sentence[i].lower()
                sentence[i] = word_to_idx.get(word, unk_idx)
            sentence.insert(0, s_idx)
            sentence.append(s_idx)


with open('data/mscoco/dict.pckl', 'wb') as f:
    cPickle.dump({'word_to_idx': word_to_idx,
                  'idx_to_word': idx_to_word}, f)

with open('data/mscoco/train_captions.json', 'w') as f:
    json.dump(train_captions, f)
np.save('data/mscoco/train_features.npy', train_features)
with open('data/mscoco/valid_captions.json', 'w') as f:
    json.dump(valid_captions, f)
np.save('data/mscoco/valid_features.npy', valid_features)
with open('data/mscoco/test_captions.json', 'w') as f:
    json.dump(test_captions, f)
np.save('data/mscoco/test_features.npy', test_features)