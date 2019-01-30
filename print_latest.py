#import matplotlib.pyplot as plt
import sys
import pickle
import os

assert len(sys.argv) == 3, 'Input dumps dir/model id, file id'
dir = sys.argv[1]
model_id = sys.argv[1].split('/')[1]
file_id = 'losses_meters' if 'loss' in sys.argv[2] else ('accuracy_meters' if 'acc' in sys.argv[2] else None)

#01_25_08_07_2_losses_meters.p

# Retrieve latest epoch
file_names = ['{}/{}'.format(dir,f) for f in os.listdir(dir) if 'model' in f]
file_names.sort(key=os.path.getctime)

latest_model_file_name = file_names[-1]
last_underscore = latest_model_file_name.rfind('_')
second_last_underscore = latest_model_file_name[:last_underscore].rfind('_')
epoch = int(latest_model_file_name[second_last_underscore+1:last_underscore])

print('Epoch {}'.format(epoch))

train_file_name = '{}/{}_{}_{}.p'.format(dir, model_id, epoch, file_id)
val_file_name = '{}/{}_{}_eval_{}.p'.format(dir, model_id, epoch, file_id)

train_data = pickle.load(open(train_file_name,'rb'))
val_data = pickle.load(open(val_file_name,'rb'))

if 'messages' in file_id:
    print('Train')
    print(train_data)
    print('Val')
    print(val_data)
else:
    print('Train', [p.avg for p in train_data])
    print('Val', [p.avg for p in val_data])
