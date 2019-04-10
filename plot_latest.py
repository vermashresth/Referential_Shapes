import matplotlib.pyplot as plt
import sys
import pickle
import os

assert len(sys.argv) == 3, 'Input dumps folder/model id, file id'
folder = sys.argv[1]
model_id, _, _ = folder.split('/')[1].split('_')
file_id = 'losses_meters' if 'loss' in sys.argv[2] else ('accuracy_meters' if 'acc' in sys.argv[2] else None)

#01_25_08_07_92_losses_meters.p # Not anymore
#0314200052021692_46_accuracy_meters.p

# Retrieve latest epoch
file_names = ['{}/{}'.format(folder,f) for f in os.listdir(folder) if '{}.p'.format(file_id) in f]
file_names.sort(key=os.path.getctime)

# print(file_names)
assert len(file_names) == 2 and 'eval' in file_names[-1]

train_file, val_file = file_names

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
fig_id = '{}'.format(model_id)
output_file_name = '{}/{}_curves_{}.png'.format(folder, file_id, fig_id)
plt.savefig(output_file_name)
print('Plot saved in file {}'.format(output_file_name))