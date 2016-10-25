import sys
import time
from tqdm import tqdm
from utils import *
from datetime import datetime
from gru_theano import GRUTheano

# SGD Callback function
def sgdCallback(model, num_examples):
  dt = datetime.now().isoformat()
  loss = model.calculateLoss(x_train[:10000], y_train[:10000])
  log_file.write('%s (%d)\n' % (dt, num_examples))
  log_file.write('-----------------------------------------------------\n')
  examples = generateGuns(model, 3, index_to_word, word_to_index)
  log_file.write('\n'.join(examples))
  log_file.write('\nSaving model..\n')
  saveModelParams(model, word_to_index, index_to_word, model_output_file)
  log_file.write('Save Complete!\n\n')

# Script Data
learning_rate = .001
vocab_size = 5000
embedding_dim = 48
hidden_dim = 128
nepoch = 20
model_output_file = "GRU-%s.dat" % datetime.now().strftime('%Y-%m-%d-%H-%M')
input_data_file = ['archive_0.zip']
print_every = 25000
log_file = open('log.txt','w')

# Load in data
print 'Constructing training data...'
x_train, y_train, word_to_index, index_to_word = loadData(input_data_file, vocab_size)
print 'Training Data Assembly Complete!\n'

# Build Model
model = GRUTheano(vocab_size, hidden_dim=hidden_dim, bptt_truncate=-1)
saveModelParams(model, word_to_index, index_to_word, model_output_file)

# Do one SGD step and print time for a step
print 'Performing one SGD step...'
t1 = time.time()
model.sgdStep(x_train[10], y_train[10], learning_rate)
t2 = time.time()
print 'Complete!\nSGD Step Time: %f milliseconds' % ((t2 -t1)/1000.0)

# Begin training
print 'Beginning Training over %d epochs...' % nepoch
for epoch in tqdm(range(nepoch),desc='Epochs'):
  trainWithSGD(model, x_train, y_train, learning_rate=learning_rate,
               nepoch=1, decay=.9, callback_every=print_every,
               callback=sgdCallback)

# Training is complete! Save trained model
print '\nTraining Complete!\nSaving trained model...'
saveModelParams(model, word_to_index, index_to_word, model_output_file)
print 'Saving complete!\nDone!'
