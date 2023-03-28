import glob, os
# parsing valid sentence (txt,wav) pairs
txt=list(glob.glob('*.txt'))
texts=[open(fn).read() for tfn in txt]
texts=[open(tfn).read() for tfn in txt]
sorted(len(text) for text in texts)
%hist
sents=zip(txt,texts)
good_sents=[(tfn, text) for tfn, text in sents if len(text)>50]
print(len(good_sents), len(sents))
print(len(good_sents), len(list(sents)))
len(txt)
# quedan 40 afuera (2.5%)
# quedan 40 afuera (6.5%)
from pprint import pprint
pprint([t for t in texts if len(t)<=50])
good_sents[:10]
fw=open('metadata.csv','w')
for tfn, text in good_sents:
    fw.write('%s|%s%s' %(tfn, text, '.'+chr(10)))
fw.close()
!head metadata.csv
%hist -f make_meta.py
