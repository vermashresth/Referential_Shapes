#import matplotlib.pyplot as plt
import sys
import pickle

assert len(sys.argv) > 1, 'Input dumps dir, model id, train file id, epoch'
dir = '{}/{}'.format(sys.argv[1], sys.argv[2])
model_id = sys.argv[2]
file_id = sys.argv[3]
epoch = sys.argv[4]

#01_25_08_07_2_losses_meters.p

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



#iterations = list(range(len(train_losses)))

#plt.plot(iterations, train_losses, color='blue')
#plt.plot(iterations, val_losses, color='green')
#plt.legend(('Train', 'Validation'))
#plt.xlabel('Iteration')
#plt.ylabel('Running avg loss')
#plt.title('Loss curves')
##plt.show()
#file_id = train_losses_file[train_losses_file.rfind('_'):-2]
#plt.savefig('{}/losses_curves_{}.png'.format(folder, file_id))
