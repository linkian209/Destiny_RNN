import sys
import time
from tqdm import tqdm
from tensor_utils import *
from datetime import datetime
from gru_tensor import GRUTensor

# SGD Callback function
def sgdCallback(model, num_examples, loss):
  dt = datetime.now().isoformat()
  with open(log_file, 'a'):
      log_file.write('%s (%d)\n' % (dt, num_examples))
      log_file.write('-----------------------------------------------------\n')
      log_file.write('Average Loss over %d training data: %f' % (num_examples,loss))
      log_file.write('\nSaving model..\n')
      model['saver'].save(sess, model_output_file)
      log_file.write('Save Complete!\n')
      examples = generateGuns(model, 3, index_to_word, word_to_index)
      log_file.write('\n'.join(examples))

# Script Data
learning_rate = .001
vocab_size = 256
hidden_dim = 128
nepoch = 20
model_output_file = "GRU-%s" % datetime.now().strftime('%Y-%m-%d-%H-%M')
input_data_file = ['archive_0.zip']
print_every = 25000
log_file = 'log.txt'

# Load in data
print 'Constructing training data...'
x_train, y_train, word_to_index, index_to_word = loadDataChars(input_data_file, vocab_size)
print 'Training Data Assembly Complete!\n'

# Build Model
print 'Creating model...'
model = GRUTensor(vocab_size=vocab_size, hidden_dim=hidden_dim, num_steps=len(max(x_train)))
print 'Model created'

# Do one SGD step and print time for a step
print 'Performing one SGD step...'
t1 = time.time()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    sess.run([model['total_loss'],
              model['final_state'],
              model['train_step']],
              {model['x']: x_train[10:42], model['y']: y_train[10:42]})
t2 = time.time()
print 'Complete!\nStep Time + variable creation: %f milliseconds' % ((t2 -t1) * 1000.0)

# Begin training
print 'Beginning Training over %d epochs...' % nepoch
losses = train(model, x_train, y_train, learning_rate=learning_rate,
               nepoch=20, callback_every=print_every,
               callback=sgdCallback)

# Training is complete! Save trained model
print '\nTraining Complete!\nSaving trained model...'
model['saver'].save(sess, model_output_file)
print 'Saving complete!\nDone!'
