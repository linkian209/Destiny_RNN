import pickle
import sys
import os
import itertools
import nltk
import time
import operator
import io
import array
import zipfile
import numpy as np
from gru_theano import GRUTheano
from tqdm import tqdm
from pprint import pprint
from datetime import datetime
from decimal import Decimal

# Globals
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'
BEGIN_TOKEN = 'BEGIN_GUN'
END_TOKEN = 'END_GUN'


# makeGrid
# This function creates the iterable grid from the Destiny Manifest
# for an inputed talentGrid
def makeGrid(grid):
   retval = {}
   for item in grid['nodes']:
      col = item['column']
      row = item['row']
      if col not in retval:
         retval[col] = {}
      if row not in retval[col]:
         retval[col][row] = []
      retval[col][row].append(item['steps'])
   retval = getRolls(retval)
   return(retval)

# convert
# This function converts a list or dict of strings to utf-8
def convert(obj):
    if isinstance(obj, dict):
        return {convert(key): convert(value) for key, value in obj.iteritems()}
    elif isinstance(obj, list):
        return [convert(element) for element in obj]
    elif isinstance(obj, unicode):
        return obj.encode('utf-8')
    else:
        return obj

# getRolls
# This function takes iterable grids from makeGrid and makes a dict of
# all possible rolls for a gun
def getRolls(grid):
   retval = {}
   for col in grid:
      retval[col] = {}
      for row in grid[col]:
         retval[col][row] = getPerksByRow(grid[col][row])
   return(retval)

# getPerksByRow
# Recursive function to get all perks for an inputted row on the gun's
# talent grid
def getPerksByRow(row):
   retval = []
   def loop(row):
      for i in row:
         if not isinstance(i, list):
            retval.append(i['nodeStepName']+'&split;'+i['nodeStepDescription'])
         else:
            loop(i)
   loop(row)
   return retval

# makeTrainingData
# This function takes the dict created in getRolls and formats it for use
# in training the RNN
def makeTrainingData(items, stats, perks, grid, item_hash):
   # Stat hashes [    ROF   ,   IMPACT  ,    RANGE  , STABILITY,   RELOAD  ]
   stat_hashes = [4284893193, 4043523819, 1240592695, 155624089, 4188031367]
   filename = '%d.txt' % item_hash
   with open(filename,'w') as f:
      combos = []
      item = items[item_hash]
      # Creates the Cartesian Product off all the rolls
      for i in grid:
         combos.append([v for v in itertools.product(*grid[i].values())])
      rolls = [v for v in itertools.product(*combos)]
      # Start iterating through and output to file
      # Section identifiers will be in all caps to help the network
      # understand what is what
      for cur_roll in tqdm(rolls, desc=unicode(item['itemName'],'utf-8')):
         roll = 'BEGIN_GUN NAME_BEGIN %s END_NAME ' % item['itemName']
         roll += 'DESCRIPTION_BEGIN %s END_DESCRIPTION ' % item['itemDescription']
         roll += 'TIER_BEGIN %s END_TIER ' % item['tierTypeName']
         roll += 'TYPE_BEGIN %s END_TYPE ' % item['itemTypeName']
         # Do the stats now
         roll += 'STATS_BEGIN '
         for stat_hash in item['stats']:
            try:
               stat = stats[int(stat_hash)]
               roll += '%s_BEGIN' % stat['statName'].replace(' ','_').upper()
               roll += ' %d ' % item['stats'][stat_hash]['value']
               roll += 'END_%s ' % stat['statName'].replace(' ','_').upper()
            except:
               continue
         roll += 'END_STATS '
         # Now do perks
         roll += 'PERKS_BEGIN '
         col_id = 0
         for col in cur_roll:
            # Start the Column
            row_id = 0
            roll += 'COL%d_BEGIN ' % col_id
            # Do individual rows
            for row in col:
               roll += 'ROW%d_BEGIN ' % row_id
               roll += ('PERK_NAME_BEGIN %s END_PERK_NAME '
                        'PERK_DESC_BEGIN %s END_PERK_DESC ') % tuple(row.replace('\n',' ').split('&split;'))
               roll += "END_ROW%d " % row_id
               row_id += 1
            roll += 'END_COL%d ' % col_id
            col_id += 1
         # Done with perks. Finish roll then right it out!
         roll += 'END_PERKS END_GUN\n'
         f.write(roll)

# makeTrainingDataJSON
# This function takes the dict created in getRolls and formats it for use
# in training the RNN in JSON form
def makeTrainingDataJSON(items, stats, perks, grid, item_hash):
   filename = '%d.txt' % item_hash
   with open(filename,'w') as f:
      combos = []
      item = items[item_hash]
      # Creates the Cartesian Product off all the rolls
      for i in grid:
         combos.append([v for v in itertools.product(*grid[i].values())])
      rolls = [v for v in itertools.product(*combos)]
      # Start iterating through and output to file
      # Section identifiers will be in all caps to help the network
      # understand what is what
      for cur_roll in tqdm(rolls, desc=unicode(item['itemName'],'utf-8')):
         roll = '{\"name\":\"%s\",' % item['itemName']
         roll += '\"description\": \"%s\",' % item['itemDescription'].replace("\"", "\\\"")
         roll += '\"tier\": \"%s\",' % item['tierTypeName']
         roll += '\"type\": \"%s\",' % item['itemTypeName']
         # Do the stats now
         roll += '\"stats\":{'
         statstrs = []
         for stat_hash in item['stats']:
            try:
               stat = stats[int(stat_hash)]
               statstr = '\"%s\":' % stat['statName'].lower()
               statstr += '%d' % item['stats'][stat_hash]['value']
               statstrs.append(statstr)
            except:
               continue
         roll += ','.join(statstrs)
         roll += '},'
         # Now do perks
         roll += '\"perks\":{'
         col_id = 0
         colstrs = []
         for col in cur_roll:
            # Start the Column
            row_id = 0
            colstr = '\"%d\":{' % col_id
            rowstrs = []
            # Do individual rows
            for row in col:
               rowstr = '\"%d\":[ ' % row_id
               rowstr += ('\"%s\",'
                          '\"%s\"') % tuple(row.replace('\n',' ').split('&split;'))
               rowstr += ']'
               rowstrs.append(rowstr)
               row_id += 1
            colstr += ','.join(rowstrs)
            colstr += '}'
            col_id += 1
            colstrs.append(colstr)
         # Done with perks. Finish roll then right it out!
         roll += ','.join(colstrs)
         roll += '}}\n'
         f.write(roll)

# init
# This function is used for testing to auto initialize several components
# necessary to create training data
def init():
   with open('manifest.pickle','rb') as f:
      data = pickle.loads(f.read())

   data = convert(data)
   items = data['DestinyInventoryItemDefinition']
   grids = data['DestinyTalentGridDefinition']
   all_items = {}
   for i in items:
      if 'itemName' in items[i].viewkeys():
         all_items[items[i]['itemName']] = {'grid': items[i]['talentGridHash'], 'hash': i}
   return(data,items,grids,all_items)

# getLines
# Generator to get lines from an inputted file
def getLines(f, max_rolls):
   num_got = 0
   # Use probabilistic sampling to get around max_rolls lines
   for line in f:
      if np.random.rand() > (num_got / (max_rolls+0.0)):
         num_got += 1
         yield line.replace('\n','').split(' ')

# getLinesChars
# Generator to get lines from an inputted file as a list of chars
def getLinesChars(f, max_rolls):
   num_got = 0
   # Use probabilistic sampling to get around max_rolls lines
   for line in f:
      if np.random.rand() > (num_got / (max_rolls+0.0)):
         num_got += 1
         yield [BEGIN_TOKEN]+[x for x in line]+[END_TOKEN]

# loadData
# Takes in a list of archive file names and loads in the training data
def loadData(filenames,vocab_size=2000, max_rolls=1000):
   # Initialize Vars
   word_to_index = []
   index_to_word = []
   vocab = []
   tokenized = []

   # Read in data
   print 'Beginning data load...'
   for cur_archive in tqdm(filenames,desc="Archives"):
      # Get the contents of the current archive
      zf = zipfile.ZipFile(cur_archive, 'a', zipfile.ZIP_DEFLATED, allowZip64=True)
      zf.extract('contents.txt')
      with open('contents.txt','r') as f:
         files = f.read().split('\n')

      files = [x for x in files if x]

      # Loop through files and read the data
      for cur_file in tqdm(files, desc="Files"):
         # Extract the current file
         zf.extract(cur_file)
         # Get the first 100 rolls and tokenize them
         num_rolls = 0
         with open(cur_file, 'r') as f:
            tokenized += list(getLines(f, max_rolls))
         # Delete the current file
         os.remove(cur_file)

      # After looping through this archive's files, delete the contents.txt
      # and start on the next one
      os.remove('contents.txt')

   print '\nData load complete!'

   # After looping through the data, get the most used words and use them
   # as our vocab
   print 'Creating vocab...'
   word_freq = nltk.FreqDist(itertools.chain(*tokenized))
   vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]))
   vocab = sorted(vocab, key=operator.itemgetter(1))
   print 'Vocab Created!'

   print 'Creating Indices...'
   index_to_word = ["<MASK/>", UNKNOWN_TOKEN] + [x[0] for x in tqdm(vocab, desc='Index to Word')]
   word_to_index = dict([(w,i) for i,w in tqdm(enumerate(index_to_word), desc='Word to Index')])
   print 'Indices Created!'

   # Replace missing words with the unknown token
   print 'Checking for unknown tokens....'
   for i, sent in tqdm(enumerate(tokenized), desc='Tokenized'):
      tokenized[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]
   print 'Complete!'

   # Create Training Data Arrays
   print 'Creating training arrays...'
   x_train = np.asarray([[word_to_index[w] for w in tqdm(sent[:-1], desc='Words')] for sent in tqdm(tokenized,desc='x_train')],dtype='int32')
   y_train = np.asarray([[word_to_index[w] for w in tqdm(sent[1:], desc='Words')] for sent in tqdm(tokenized, desc='y_train')],dtype='int32')
   print '\nTraining arrays created!'

   return x_train, y_train, word_to_index, index_to_word

# loadDataChars
# Takes in a list of archive file names and loads in the training data as
# lists of characters
def loadDataChars(filenames,vocab_size=128, max_rolls=1000, max_len=2000):
   # Initialize Vars
   word_to_index = []
   index_to_word = []
   vocab = []
   tokenized = []

   # Read in data
   print 'Beginning data load...'
   for cur_archive in tqdm(filenames,desc="Archives"):
      # Get the contents of the current archive
      zf = zipfile.ZipFile(cur_archive, 'a', zipfile.ZIP_DEFLATED, allowZip64=True)
      zf.extract('contents.txt')
      with open('contents.txt','r') as f:
         files = f.read().split('\n')

      files = [x for x in files if x]

      # Loop through files and read the data
      for cur_file in tqdm(files, desc="Files"):
         # Extract the current file
         zf.extract(cur_file)
         # Get the first 100 rolls and tokenize them
         num_rolls = 0
         with open(cur_file, 'r') as f:
            tokenized += list(getLinesChars(f, max_rolls))
         # Delete the current file
         os.remove(cur_file)

      # After looping through this archive's files, delete the contents.txt
      # and start on the next one
      os.remove('contents.txt')

   print '\nData load complete!'
   print 'Padding examples...'
   for sent in tqdm(tokenized, desc="Examples"):
       if len(sent) < max_len:
           while len(sent) <= max_len:
               sent.append(' ')
   print 'Padding complete!'

   # After looping through the data, get the most used words and use them
   # as our vocab
   print 'Creating vocab...'
   word_freq = nltk.FreqDist(itertools.chain(*tokenized))
   vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]))
   vocab = sorted(vocab, key=operator.itemgetter(1))
   print 'Vocab Created!'

   print 'Creating Indices...'
   index_to_word = ["<MASK/>", UNKNOWN_TOKEN] + [x[0] for x in tqdm(vocab, desc='Index to Word')]
   word_to_index = dict([(w,i) for i,w in tqdm(enumerate(index_to_word), desc='Word to Index')])
   print 'Indices Created!'

   # Replace missing words with the unknown token
   print 'Checking for unknown tokens....'
   for i, sent in tqdm(enumerate(tokenized), desc='Tokenized'):
      tokenized[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]
   print 'Complete!'

   # Create Training Data Arrays
   print 'Creating training arrays...'
   x_train = np.asarray([[word_to_index[w] for w in tqdm(sent[:-1], desc='Words')] for sent in tqdm(tokenized,desc='x_train')],dtype='int32')
   y_train = np.asarray([[word_to_index[w] for w in tqdm(sent[1:], desc='Words')] for sent in tqdm(tokenized, desc='y_train')],dtype='int32')
   print '\nTraining arrays created!'

   return x_train, y_train, word_to_index, index_to_word

# trainWithSGD
# Take the inputted model and training data and run a number of epochs
# over the data to train the model
def trainWithSGD(model, x_train, y_train, learning_rate=.001, nepoch=20,
  decay=0.9, callback_every=10000, callback=None, batch_size=32):
  # Keep track of number examples seen for the callback
  examples_seen = 0
  total_examples = len(x_train)
  num_batches = np.ceil(1. * total_examples / batch_size)
  callback_freq = np.ceil(callback_every / batch_size)
  # Loop through epochs
  for epoch in tqdm(range(nepoch), desc='Epochs'):
    for i in tqdm(range(int(num_batches)), desc='Batches'):
      # Batch Params
      batch_start = i * batch_size
      batch_end = min(total_examples, (i+1) * batch_size)
      # If we don't have enough in the epoch to do a whole batch,
      # repeat some from the beginning
      if (batch_end == total_examples) and (batch_end - batch_start < batch_size):
          remaining = batch_end - batch_start
          train_x = x_train[batch_start:batch_end] + x_train[0:remaining]
          train_y = y_train[batch_start:batch_end] + y_train[0:remaining]
          # Do the last step
          model.sgdStep(train_x, train_y, learning_rate, decay)
      else:
		  # Do one step with Stochastic Gradient Descent
		  model.sgdStep(x_train[batch_start:batch_end],
		                y_train[batch_start:batch_end], learning_rate, decay)
      examples_seen += (batch_end - batch_start)
      # Do the callback if we have a callback and have seen enough
      if(callback and callback_every and
        (examples_seen % callback_freq == 0 or (i+1) > num_batches)):
        # Do callback
        callback(model, examples_seen)

  # Return the model once we complete training
  return model

# saveModelParams
# Takes an inputted model and filename and outputs the model to file
def saveModelParams(model, word_to_index, index_to_word, outfile):
  np.savez(outfile,
    E=model.E.get_value(),
    U=model.U.get_value(),
    W=model.W.get_value(),
    V=model.V.get_value(),
    b=model.b.get_value(),
    c=model.c.get_value(),
    word_to_index=word_to_index,
    index_to_word=index_to_word)

  print "Saved model to %s!" % outfile

# loadModelParams
# Takes in a path to a file, loads in the data, and builds a model from
# the saved data
def loadModelParams(path, modelClass=GRUTheano, indexes=True):
  # Load data and params
  npzfile = np.load(path)
  E, U, W, V = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"]
  b, c = npzfile["b"], npzfile["c"]
  hidden_dim, word_dim = E.shape[0], E.shape[1]

  # Build Model
  print "Building model..."
  model = modelClass(word_dim, hidden_dim=hidden_dim)
  model.E.set_value(E)
  model.U.set_value(U)
  model.W.set_value(W)
  model.V.set_value(V)
  model.b.set_value(b)
  model.c.set_value(c)
  print "Complete!"

  if indexes:
    return model, npzfile['word_to_index'].item(), npzfile['index_to_word'].tolist()
  else:
    return model

# gradientCheckTheano
# Takes the inputted model and training data to check the model's parameters
# for error
def gradientCheckTheano(model, x, y, h=0.001, error_threshold=0.01):
  # Overwrite truncate value so we can backpropgate all the way
  model.bptt_truncate = 1000
  # Now calculate backpropation gradients
  bptt_gradients = model.bptt(x,y)

  # Perform gradient check for each parameter we want to check
  model_params = ['E', 'U', 'W', 'V', 'b', 'c']
  for idx, name in tdqm(enumerate(model_params), desc='Params'):
    # Get actual param from model
    param_T = operator.attrgetter(name)(model)
    param = param_T.get_value()

    # Iterate over each element of the param matrix
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
      i = it.multi_index
      # Save copy of original param
      original = param[i]
      # Estimate gradient using central difference formula
      param[i] = original + h
      param_T.set_value(param)
      grad_plus = model.calculateTotalLoss([x],[y])
      param[i] = original - h
      param_T.set_value(param)
      grad_minus = model.calculateTotalLoss([x],[y])
      estimated = (grad_plus - grad_minus) / (2 * h)
      # Set param back to original
      param_T.set_value(original)
      # Get gradient calculated through BPTT
      bp_grad = bptt_gradients[idx][i]
      # Calculate Relative Error
      rel_err = np.abs(bp_grad - estimated) / (np.abs(bp_grad) + np.abs(estimated))
      # Check the error to make sure it is within spec
      if rel_err > error_threshold:
        print "Gradient Check Error: param=%s, i=%s" % (name, i)
        print "+h loss: %f" % grad_plus
        print "-h loss: %f" % grad_minus
        print "Estimated gradient: %f" % estimated
        print "Backpropogation gradient: %f" % bp_grad
        print "Relative error: %f" % rel_err
        return

      it.iternext()

    print "Gradient check for param %s passed!" % name

# generateGun
# Takes in the trained model and the word indices and returns a gun
def generateGun(model, index_to_word, word_to_index, min_length=20):
  # Start with the begin token
  new_gun = [word_to_index[BEGIN_TOKEN]]
  # Repeat until we get an end token or the gun is too long (>300 words)
  # to make sure we aren't getting caught in some loop
  while not new_gun[-1] == word_to_index[END_TOKEN]:
    next_word_prob = model.predict(new_gun)[-1]
    # Normalize prediction output
    if sum(next_word_prob) > 1:
        next_word_prob = np.divide(next_word_prob, sum(next_word_prob))
    # Loop to make sure we don't sample an unknown token
    sampled_word = word_to_index[UNKNOWN_TOKEN]
    while sampled_word == word_to_index[UNKNOWN_TOKEN]:
        print next_word_prob
        print index_to_word[sampled_word]
        samples = np.random.multinomial(1, next_word_prob)
        sampled_word = np.argmax(next_word_prob)
    # Add this to the new gun
    new_gun.append(sampled_word)
    # Make sure we aren't stuck
    if len(new_gun) > 2000:
      return None

  if len(new_gun) < min_length:
    return None

  return [index_to_word(w) for w in new_gun]

# generateGuns
# Given a model, word indices, and a number of guns to make, this function
# generates guns from the model
def generateGuns(model, n, index_to_word, word_to_index, filename=None):
  # retval is a list of guns
  retval = []
  # If we did not get a filename, make a default one
  if not filename:
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = 'guns_%s.txt' % time

  with open(filename, 'w') as f:
    for i in tqdm(range(n), desc="Guns"):
      sent = None
      while not sent:
        sent = generateGun(model, index_to_word, word_to_index)

      pprint(sent)
      f.write(" ".join(sent))
      f.write('\n')
      retval.append(' '.join(sent))

  return retval
