import sys
import os
import pickle

assert len(sys.argv) == 2, 'You need a model id'

model_id = sys.argv[1]

dumps_dir = '../dumps'
model_dir = '{}/{}'.format(dumps_dir, model_id)

messages_file_name = ['{}/{}'.format(model_dir, f) for f in os.listdir(model_dir) if 'test_messages.p' in f]

assert len(messages_file_name) == 1
messages_file_name = messages_file_name[0]

# Load messages
x = pickle.load(open(messages_file_name, 'rb'))
print(x[0])


# Grab stats

