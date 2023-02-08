# duration = 46 mins
#import glob
#from pydub import AudioSegment
#audios=[AudioSegment.from_file(fn) for fn in glob.glob('*')]
#print('MINUTES=', sum([au.duration_seconds for au in audios])/60)

import boto3

s3 = boto3.client('s3')
transcribe = boto3.client('transcribe')

bucket_name = 'quantcldata'
folder_name = 'TITOVOX/CHOPPED'

objects = s3.list_objects(Bucket=bucket_name, Prefix=folder_name)
print('nOBJECTS:', len(objects['Contents']) # should be 39

# Loop through the objects in the S3 folder

for s3_object in objects['Contents']:   # could be done async

    file_name = s3_object['Key']
    media_format = file_name[-3:]

    try:
        transcribe_response = transcribe.start_transcription_job(
            TranscriptionJobName=file_name,
            LanguageCode='es-MX',   # perhaps es-ES?
            MediaFormat=media_format,
            Media={
                'MediaFileUri': f's3://{bucket_name}/{file_name}'
            }
        )
        head = f'Started transcription job for {file_name}'
        tail = transcribe_response["TranscriptionJob"]["TranscriptionJobName"]
        print(f'{head} with job name {tail}')
    except:
        print('COULD NOT HANDLE:', file_name)

