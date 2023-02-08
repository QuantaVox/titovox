** ETL steps **

1. upload files from a public folder (eg: Google Drive) to PREP_DATA
2. copy files into a Colab (copy it here)
3. extract mp3 from YT_URL, or video files
4. upload mp3s to s3 (do this with a script from the cmd line)
5. send mp3s to be AWS Transcribed
6. recover output JSON files
7. construct sentences=(.mp3, .txt)
8. upload to folder TRAINING_DATA

TO DO: clip files using filename.mp3 + filename.ts (Tito speaks = (start,stop) timestamps)
