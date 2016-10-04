# encoding=utf8

import utils
import pickle
import zipfile
import os
from tqdm import tqdm
from pprint import pprint

# Globals
#[Special, Heavy, Primary]
bucketHashes = [2465295065,953998645,1498876634]

# Load in Manifest
print 'Loading Manifest...'
with open('manifest.pickle','rb') as f:
   data = pickle.loads(f.read())

# Convert strings to Unicodie
print 'Converting Manifest...'
data = utils.convert(data)

# Get the Items, Grid, Stats, and Perks tables from the Manifest
items = data['DestinyInventoryItemDefinition']
grids = data['DestinyTalentGridDefinition']
stats = data['DestinyStatDefinition']
perks = data['DestinySandboxPerkDefinition']

# Get all named items from the database
all_items = {}
print 'Creating items....\n'
for i in tqdm(items, desc='Item Gathering'):
   # Get Weapons
   if items[i]['bucketTypeHash'] in bucketHashes:
      if 'itemName' in items[i].viewkeys():
         all_items[items[i]['itemName']] = {'grid':items[i]['talentGridHash'],'hash': i}

# Loop through items and create training data
cur_arch = 0
num_guns = 0
hash_list = []
bad_hashes = []
print '\nLooping through Guns to create training data...\n'
for item in tqdm(all_items, desc='Guns'):
   gun = all_items[item]
   cur_archive = 'archive_%d.zip' % cur_arch
   # First check to see if this archive exists, if not make it
   if not os.path.exists(cur_archive):
      zf = zipfile.ZipFile(cur_archive, 'a', zipfile.ZIP_DEFLATED, allowZip64=True)
      zf.close()
   # Make sure this archive can handle another file
   if not(os.stat(cur_archive).st_size <= 3900000000):
      # Create a contents file for the archive
      with open('contents.txt','w') as f:
         for i in hash_list:
            f.write('%d.txt' % i)
      zf = zipfile.ZipFile(cur_archive, 'a', zipfile.ZIP_DEFLATED, allowZip64=True)
      zf.write('contents.txt')
      zf.close()
      os.remove('contents.txt')
      cur_arch += 1
      hash_list = []

   # Open zipfile
   zf = zipfile.ZipFile(cur_archive, 'a', zipfile.ZIP_DEFLATED, allowZip64=True)

   # Create grid for gun
   # If it is no good, just continue onto the next
   try:
      grid = utils.makeGrid(grids[gun['grid']])
   except:
      bad_hashes.append(gun['hash'])
      continue

   # Create the training data! 
   utils.makeTrainingData(items, stats, perks, utils.makeGrid(grids[gun['grid']]), gun['hash'])

   # Add this to the zipfile
   zf.write('%d.txt' % gun['hash'])
   zf.close()

   # Remove the file and add the hash to the list
   os.remove('%d.txt' % gun['hash'])
   hash_list.append(gun['hash'])
   num_guns += 1

# Done! Add contents to the last archive
with open('contents.txt','w') as f:
   for i in hash_list:
      f.write('%d.txt\n' % i)

zf = zipfile.ZipFile('archive_%d.zip' % cur_arch, 'a', zipfile.ZIP_DEFLATED, allowZip64=True)
zf.write('contents.txt')
zf.close()
os.remove('contents.txt')

# Show completion and print end stats!
print '\nComplete!'
print 'Created training data for %d guns across %d %s!' % (num_guns, cur_arch+1, 'archives' if cur_arch > 0 else 'archive')
print 'Skipped %d hashes!' % len(bad_hashes)
