## Update Julio UC Davis
1. se revisó la segmentación de las frases usando AWS Transcribe
2. con esto se construyó un conjunto (WAV,TXT)
3. ahora intentaremos nuevamente con FakeYou

## 08-marzo (TV-7): Resumen Pipeline
0. Lucas upload to Drive -> manual Drive2git -> PREP_DATA
1. PREP_DATA -> manual git2s3 -> s3://quantcldata/titovox
2. s3 -> scripts/s3_transcriber.py -> AWS Transcribe jobs
3. Finished jobs -> S3 output buckets (one needs to wait)
4. aws s3 sync s3://titovox-transcripts 

TO DO: view as an Airflow/Bonobo graph
IDEALLY:   one should be able to git clone this repo and run a single script that submits all files inside a given folder to AWS.

## ETL steps 

6. recover output JSON files
7. construct sentences=(.mp3, .txt)
8. upload to folder TRAINING_DATA

TO DO: clip files using filename.mp3 + filename.ts (Tito speaks = (start,stop) timestamps)

## DONE
1. Lucas subió archivos a un Drive. Sergio copió estos a este Github (PREP_DATA)
75Mb in 39 files (4 audio formats): {'m4a': 26, 'mp4': 9, 'aac': 2, 'ogg': 2}
2. Sergio transformó todos estos a mp3 usando AudioSegment (scripts/audio_transformer.py).
3. YouTube SKIP (for now)
4. Sergio subió estos archivos manualmente a una carpeta en s3://quantcldata/titovox (46 minutos)
5. chatGPT me enseñó como subir una carpeta a AWS Transcribe (scripts/s3_transcriber.py)
6. how many files end up in s3://titovox-transcripts? 
   
   aws s3 sync s3://titovox-transcripts TRANSCRIPTS
