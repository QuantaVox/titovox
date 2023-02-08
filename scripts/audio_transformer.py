import glob, os
from pydub import AudioSegment

audios=[AudioSegment.from_file(fn) for fn in glob.glob('PREP_DATA/*/*')]
files=glob.glob('PREP_DATA/*/*') # list not needed
OUT='CHOPPED_DATA'

for file, audio in zip(files,audios):
    audio.export(os.path.join(OUT, file.split('/')[-1].replace(' ','_')), format='mp3')
    print(file)
