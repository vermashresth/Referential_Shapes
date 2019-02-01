import matplotlib.pyplot as plt
import sys
import pickle
import os

assert len(sys.argv) == 3, 'Input dumps folder/model id, file id'
dumps_folder, model_id = sys.argv[1].split('/')
file_id = 'losses_meters' if 'loss' in sys.argv[2] else ('accuracy_meters' if 'acc' in sys.argv[2] else None)

#01_25_08_07_92_losses_meters.p

folder = '{}/{}'.format(dumps_folder, model_id)

# Retrieve latest epoch
file_names = ['{}/{}'.format(folder,f) for f in os.listdir(folder) if 'model' in f]
file_names.sort(key=os.path.getctime)

latest_model_file_name = file_names[-1]
last_underscore = latest_model_file_name.rfind('_')
second_last_underscore = latest_model_file_name[:last_underscore].rfind('_')
epoch = int(latest_model_file_name[second_last_underscore+1:last_underscore])


train_file = '{}/{}_{}_{}.p'.format(folder, model_id, epoch, file_id)
val_file = '{}/{}_{}_eval_{}.p'.format(folder, model_id, epoch, file_id)

if not os.path.exists(train_file):
    epoch -= 1
    train_file = '{}/{}_{}_{}.p'.format(folder, model_id, epoch, file_id)
    val_file = '{}/{}_{}_eval_{}.p'.format(folder, model_id, epoch, file_id)

train_data = pickle.load(open(train_file,'rb'))
val_data = pickle.load(open(val_file,'rb'))

iterations = list(range(len(train_data)))

plt.plot(iterations, [d.avg for d in train_data], color='blue')
plt.plot(iterations, [d.avg for d in val_data], color='green')
plt.legend(('Train', 'Validation'))
plt.xlabel('Epoch')

metric = file_id[:file_id.rfind('_')]

plt.ylabel('Avg {}'.format(metric))
plt.title('{} curves'.format(metric))
#plt.show()
fig_id = '{}_{}'.format(model_id, epoch)
plt.savefig('{}/{}_curves_{}.png'.format(folder, file_id, fig_id))
