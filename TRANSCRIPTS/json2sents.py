import glob
import json
import pandas as pd
from pydub import AudioSegment

jfiles = glob.glob('JSON/*.json')
results = map(lambda fn: json.load(open(fn))['results'], jfiles)

rdf = pd.DataFrame()
jx = 0
filename = 'SENTENCES/sentence_%03d.txt' %jx
fw = open(filename, 'w')

for ix, res in enumerate(results):   # use transcripts + items

    trans = res['transcripts']
    idf = pd.DataFrame.from_records(res['items'])    # use punctuation to split

    fn = jfiles[ix]  # could just zip
    afn = fn.replace('JSON','../CHOPPED_DATA/MP3').replace('.json','.mp3').replace('_v2.mp3','')

    # read, split, save .txt + .mp3
    aud = AudioSegment.from_file(afn)
    start = 0
    txt=''

    for _, row in idf.iterrows():   # now split
        if row['type']=='pronunciation':
            stop = float(row['end_time'])   # in seconds
            txt += (row['alternatives'][0]['content']+' ')
        else: # punctuation, save and close
            fw.write(txt)
            fw.close()
            jx+=1
            print('WROTE:', filename)
            filename = 'SENTENCES/sentence_%03d.txt' %jx
            fw = open(filename, 'w')

            clip = aud[(start*1000):(stop*1000)]
            clip.export('SENTENCES/sentence_%03d.mp3' %jx)
