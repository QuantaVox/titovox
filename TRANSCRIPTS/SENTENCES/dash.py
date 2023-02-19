import glob
import streamlit as st

AUD = '.wav'
TXT = '.txt'

for fn in glob.glob('sent*.wav'):  # could be wav

    st.audio(fn, format="audio/wav")
    text = open(fn.replace(AUD,TXT)).read()
    st.text(text)
