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
import tensorflow as tf
from tqdm import tqdm
from pprint import pprint
from datetime import datetime
from decimal import Decimal
from gru_tensor import GRUTensor

# Globals
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'
NULL_TOKEN = 'NULL_TOKEN'
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
               sent.append(NULL_TOKEN)
   print 'Padding complete!'

   # After looping through the data, get the most used words and use them
   # as our vocab
   print 'Creating vocab...'
   vocab = set()
   for sent in tokenized:
      for c in sent:
	     vocab.add(c)
   vocab.add(UNKNOWN_TOKEN)
   print 'Vocab Created!'

   print 'Creating Indices...'
   index_to_word = dict(enumerate(vocab))
   word_to_index = dict(zip(index_to_word.values(), index_to_word.keys()))
   print 'Indices Created!'

   # Replace missing words with the unknown token
   print 'Checking for unknown tokens....'
   for i, sent in tqdm(enumerate(tokenized), desc='Tokenized'):
      tokenized[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]
   print 'Complete!'

   # Create Training Data Arrays
   print 'Creating training arrays...'
   x_train = [[word_to_index[w] for w in tqdm(sent[:-1], desc='Words')] for sent in tqdm(tokenized, desc='x_train')]
   y_train = [[word_to_index[w] for w in tqdm(sent[1:], desc='Words')] for sent in tqdm(tokenized, desc='y_train')]
   print '\nTraining arrays created!'

   return x_train, y_train, word_to_index, index_to_word

# train
# Take the inputted model and training data and run a number of epochs
# over the data to train the model
def train(model, x_train, y_train, word_to_index,
                 index_to_word, model_output_file, learning_rate=.0001,
                 nepoch=20, callback_every=10000,
                 callback=None, batch_size=32):
  with tf.Session() as sess:
    # Initialize
    sess.run(tf.initialize_all_variables())
    examples_seen = 0
    total_examples = len(x_train)
    num_batches = np.ceil(1. * total_examples / batch_size)
    callback_freq = np.ceil(callback_every / batch_size)
    training_losses = []
    # Loop through epochs
    for epoch in tqdm(range(nepoch), desc='Epochs'):
      # Epoch variables
      training_loss = 0
      training_state = None
      for i in tqdm(range(int(num_batches)), desc='Batches'):
        # Batch Params
        batch_start = i * batch_size
        batch_end = min(total_examples, (i+1) * batch_size)
        # If we are at the end of the training data, 
        # wrap around and reuse
        if batch_end == total_examples:
          remaining = batch_size - (batch_end - batch_start) + 1
          train_x = x_train[batch_start:batch_end-1] + x_train[0:remaining]
          train_y = y_train[batch_start:batch_end-1] + y_train[0:remaining]
        else:
          train_x = x_train[batch_start:batch_end]
          train_y = y_train[batch_start:batch_end]
        feed_dict = {model['x']: train_x, model['y']: train_y}
        # If we have a current state, initialize
        if training_state is not None:
          feed_dict[model['init_state']] = training_state
        # Perform this step
        loss, state, _ = sess.run([model['total_loss'],
                                   model['final_state'],
                                   model['train_step']],
                                   feed_dict)
        training_loss += loss
        examples_seen += batch_size
        # Do the callback if we have a callback and have seen enough
        if(callback and callback_every and
          (examples_seen % callback_freq == 0 or (i+1) == num_batches)):
          # Do callback
          callback(model, examples_seen, training_loss/(1.*i))
          # Save Model and generating an example gun
          model['saver'].save(sess, model_output_file)
          temp = GRUTensor(num_steps=1, batch_size=1)
          examples = generateGuns(temp, 1, model_output_file, index_to_word, word_to_index)
          model['saver'].restore(sess, model_output_file)

      # Record this epoch's loss
      training_losses.append(training_loss/(1.0 *num_batches))	

  # Return the model once we complete training
  return training_losses

# generateGun
# Takes in the trained model and the word indices and returns a gun
def generateGun(model, start, num_chars=2000, vocab_size=256): 
  # Begin TF Session
  with tf.Session() as sess:
    # Start with the begin token
    state = None
    new_gun = [start]
    # Generate characters
    for i in tqdm(range(num_chars), desc='Characters'):
      # Initialize			
      if state is not None:
        feed_dict = {model['x']: [[new_gun]], model['init_state']: state}
      else:
        feed_dict = {model['x']: [[new_gun]]}

      pred, state = sess.run([model['preds'], model['final_state']], feed_dict)
      sampled = np.random.choice(vocab_size, 1, p=np.sqeeze(pred))[0]
      new_gun.append(sampled_word)
    
  return new_gun

# generateGuns
# Given a model, word indices, and a number of guns to make, this function
# generates guns from the model
def generateGuns(model, n, model_output_file, index_to_word, word_to_index, 
                 vocab_size=256, filename=None):
  # Initialize
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    model['saver'].restore(sess, model_output_file)
    print 'Loaded. Beginning Generation'

    # retval is a list of guns
    retval = []
    # If we did not get a filename, make a default one
    if not filename:
      time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
      filename = 'guns_%s.txt' % time

    with open(filename, 'w') as f:
      for i in tqdm(range(n), desc="Guns"):
         state = None
         new_gun = [word_to_index[BEGIN_TOKEN]]
         # Generate characters
         for i in range(num_chars):
            # Initialize			
            if state is not None:
              feed_dict = {model['x']: [[new_gun]], model['init_state']: state}
            else:
              feed_dict = {model['x']: [[new_gun]]}

            pred, state = sess.run([model['preds'], model['final_state']], feed_dict)
            sampled = np.random.choice(vocab_size, 1, p=np.sqeeze(pred))[0]
            new_gun.append(sampled_word)
            print new_gun

         f.write(''.join([index_to_word[x] for x in new_gun]))
         f.write('\n')
         retval.append(''.join([index_to_word[x] for x in new_gun]))

  return retval
