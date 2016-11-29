import sys
import time
import getopt
from tqdm import tqdm
from tensor_utils import *
from datetime import datetime
from gru_tensor import GRUTensor

# SGD Callback function
def sgdCallback(num_examples, loss):
  dt = datetime.now().isoformat()
  with open(log_filename, 'a')as log_file:
      log_file.write('%s (%d)\n' % (dt, num_examples))
      log_file.write('-----------------------------------------------------\n')
      log_file.write('Average Loss over %d training data: %f\n' % (num_examples,loss))

# Parse starting vars
try:
    opts, args = getopt.getopt(sys.argv[1:], 'm:e:', ['model=','epoch='])
except getopt.GetoptError as err:
    print str(err)
    print 'Expected format: python resume_tensor.py -model [model file] -epoch [starting epoch number]'
    sys.exit(2)

model_input_file = None
starting_epoch = None

for opt, arg in opts:
    if opt in ('-m', '--model'):
      model_input_file = arg
    elif opt in ('-e', '--epoch'):
      starting_epoch = int(arg)
    else:
      print 'Expected format: python resume_tensor.py -model [model file] -epoch [starting epoch number]'
      sys.exit(2)

if not starting_epoch or not model_input_file:
  print 'Missing parameter. Please input a model file and starting epoch!'
  sys.exit()

# Script Data
learning_rate = .001
vocab_size = 256
hidden_dim = 128
nepoch = 20
model_output_file = model_input_file
input_data_file = ['archive_0.zip']
print_every = 25000
log_filename = 'log.txt'

# Load in data
print 'Constructing training data...'
x_train, y_train, word_to_index, index_to_word = loadDataChars(input_data_file, vocab_size, model_file=model_output_file)
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
print 'Beginning Training over %d epochs starting on epoch %d...' % (nepoch,starting_epoch)
losses = train(model, x_train, y_train, word_to_index,
               index_to_word, model_output_file, learning_rate=learning_rate,
               nepoch=20, callback_every=print_every, load_model=True,
               callback=sgdCallback, starting_epoch=starting_epoch)

# Training is complete! Save trained model
print '\nTraining Complete!'
