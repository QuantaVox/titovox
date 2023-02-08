import glob
from pydub import AudioSegment

EXTENSIONS = ['.m4a', '.mp4']

for ext in EXTENSIONS:
    files=glob.glob('*'+ext)
    audios=map(AudioSegment.from_mp3, files)
    for file, audio in zip(files, audios):
        audio.export(file.replace(ext,'.mp3'), format='mp3')
