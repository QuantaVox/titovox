import streamlit as st
import os
import pandas as pd
import librosa

def load_data(folder):
    files = os.listdir(folder)
    data = []
    for file in files:
        if file.endswith('.txt'):
            transcription = open(os.path.join(folder, file)).read()
            audio_file = file.replace('.txt', '.wav')
            audio_path = os.path.join(folder, audio_file)
            duration = librosa.get_duration(filename=audio_path)
            data.append([file, transcription, audio_file, duration])
    df = pd.DataFrame(data, columns=['filename', 'transcription', 
'audio_file', 'duration'])
    return df

if __name__ == '__main__':
    st.title('Transcription Dashboard')
    folder = st.text_input('Enter the path to the folder:')
    if os.path.isdir(folder):
        df = load_data(folder)
        st.write('## Transcription List')
        st.write(df)
        if st.checkbox('Show audio files'):
            st.write('## Audio Files')
            for i, row in df.iterrows():
                st.write(f'{i+1}. {row["audio_file"]} ({row["duration"]:.2f} seconds)')
    else:
        st.write('Please enter a valid folder path.')
