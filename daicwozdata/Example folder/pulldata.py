import pandas as pd
import numpy as np
from pydub import AudioSegment 
import os

#VARS:
output_folder = "audioparts"
os.makedirs(output_folder, exist_ok=True)
csv = "320_TRANSCRIPT.csv"
audfile = "320_AUDIO.wav"


df = pd.read_csv(csv)
start_times = []
end_times = []
particspeaking = False
state = False
prev = [0,0,0,0]

for index, row in df.iterrows():
    rowdata = row[0].split("\t")

    #check start time:
    if rowdata[2] == "Participant":
        particspeaking = True
    else:
        particspeaking = False


    if (particspeaking != state):
        state = particspeaking
        #print(rowdata[2])
        #start time:
        if rowdata[2] == "Ellie" and prev[2] == "Participant":
            #print(prev)
            end_times.append(prev[1])
        #end time:
        if rowdata[2] == "Participant":
            start_times.append(rowdata[0])
            #print(rowdata[0])
    prev = rowdata

pairsoftimes = []
for x in range(len(end_times)):
    pairsoftimes.append((start_times[x],end_times[x]))




#print(pairsoftimes)
audio = AudioSegment.from_wav(audfile)

start = int(float(pairsoftimes[0][0])*1000)
end = int(float(pairsoftimes[0][1])*1000)
print(start," ",end)

count = 1
for x,y in pairsoftimes:
    print(x," ",y)
    start = int(float(x)*1000)
    end = int(float(y)*1000)
    audio_portion = audio[start: end]
    output_path = os.path.join(output_folder, "p{}.wav".format(count))
    audio_portion.export(output_path, format="wav")
    count += 1
