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
from tqdm import tqdm
from datetime import datetime

# Globals
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'

def makeGrid(input):
   retval = {}
   for item in input['nodes']:
      col = item['column']
      row = item['row']
      if col not in retval:
         retval[col] = {}
      if row not in retval[col]:
         retval[col][row] = []
      retval[col][row].append(item['steps'])
   retval = getRolls(retval)
   return(retval)

def convert(input):
    if isinstance(input, dict):
        return {convert(key): convert(value) for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [convert(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

def getRolls(grid):
   retval = {}
   for col in grid:
      retval[col] = {}
      for row in grid[col]:
         retval[col][row] = getPerksByRow(grid[col][row])
   return(retval)

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

def loadData(filenames,vocab_size=2000, min_sent_chars=0):
   # Initialize Vars
   word_to_index = []
   index_to_word = []
   word_freq = {}
   vocab = []

   # Read in data
   for cur_archive in tqdm(range(len(filenames)),desc="Archives"):
      # Get the contents of the current archive
      zf = zipfile.ZipFile(cur_archive, 'a', zipfile.ZIP_DEFLATED, AllowZip64=True)
      zf.extract('contents.txt')
      with open('contents.txt','r') as f:
         files = f.read().split('\n')

      # Loop through files and read the data
      for cur_file in tqdm(range(len(files)), desc="Files"):
         # Extract the current file
         zf.extract(cur_file)
         # Get the rolls from it and tokenize it
         with open(cur_file, 'r') as f:
            rolls = f.readlines()
         tokenized = [nltk.word_tokenize(roll) for roll in rolls]
         # Add it to the word frequency struct
         word_freq += nltk.FreqDist(itertools.chain(*tokenized))
         # Delete the current file
         os.remove(cur_file)

      # After looping through this archive's files, delete the contents.txt
      # and start on the next one
      os.remove('contents.txt')

   # After looping through the data, get the most used words and use them
   # as our vocab
   vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]))
   vocab = sorted(vocab, key=operator.itemgetter(1))

   index_to_word = ["<MASK/>", UNKNOWN_TOKEN] + [x[0] for x in vocab]
   word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
