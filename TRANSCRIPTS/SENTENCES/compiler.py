import glob

fw=open('index.html','w')
fw.write('<H2>Transcripciones Tito</H2>')
fw.write('<TABLE BORDER=1><TR><TH>Audio</TH><TH>Texto</TH></TR>')

for fn in glob.glob('*.txt'):
    txt=open(fn).read()
    aud=fn.replace('.txt','.mp3')
    fw.write('<TR><TD><A HREF="%s">%s</A></TD><TD>%s</TD></TR>' %(aud, aud, txt))

fw.write('</TABLE>')
fw.close()
