from pydub import AudioSegment

snd=AudioSegment.from_file('Financiamientos.mp4')
start=11.7*60*1000
stop=16.5*60*1000
extract=snd[start:stop]
extract.export('tito_vox1.wav', format='wav')
