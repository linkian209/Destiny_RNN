from urllib import request, error
from tqdm import tqdm
from pprint import pprint
import json

with open('icons.json','r') as f:
	raw = f.read()
	
icons = json.loads(raw)

num = 0
with open('log.txt','w') as log:
	for i in tqdm(icons):
		cur_icon = icons[i]
		url = 'https://www.bungie.net' + cur_icon
		try:
			req = request.Request(url)
			req.add_header('X-API-KEY','d574490e0786471c8ea20be9a803b87d')
			req = request.urlopen(req)
			filename = '%s.png' % i.replace('?','').replace('/','_').replace("\"","")
			with open(filename, 'wb') as f:
				f.write(req.read())
		except error.HTTPError as err:
			msg = url + ': ' + err.msg + '\n'
			log.write(msg)
			continue
		num += 1