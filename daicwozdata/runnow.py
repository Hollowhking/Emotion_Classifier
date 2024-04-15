import os
import time

import numpy as np
import pandas as pd

import audb
import audiofile
import opensmile
import csv
from pydub import AudioSegment 

grand = 322


#WRITE DATA TO CSV FROM MANY AUDIO FILES: ==================================
#VARS:
list_of_features = ["F0semitoneFrom27.5Hz_sma3nz_amean",
"F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
"F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
"F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
"F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
"F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
"F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
"F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
"F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",
"F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope",
"loudness_sma3_amean",
"loudness_sma3_stddevNorm",
"loudness_sma3_percentile20.0",
"loudness_sma3_percentile50.0",
"loudness_sma3_percentile80.0",
"loudness_sma3_pctlrange0-2",
"loudness_sma3_meanRisingSlope",
"loudness_sma3_stddevRisingSlope",
"loudness_sma3_meanFallingSlope",
"loudness_sma3_stddevFallingSlope",
"spectralFlux_sma3_amean",
"spectralFlux_sma3_stddevNorm",
"mfcc1_sma3_amean",
"mfcc1_sma3_stddevNorm",
"mfcc2_sma3_amean",
"mfcc2_sma3_stddevNorm",
"mfcc3_sma3_amean",
"mfcc3_sma3_stddevNorm",
"mfcc4_sma3_amean",
"mfcc4_sma3_stddevNorm",
"jitterLocal_sma3nz_amean",
"jitterLocal_sma3nz_stddevNorm",
"shimmerLocaldB_sma3nz_amean",
"shimmerLocaldB_sma3nz_stddevNorm",
"HNRdBACF_sma3nz_amean",
"HNRdBACF_sma3nz_stddevNorm",
"logRelF0-H1-H2_sma3nz_amean",
"logRelF0-H1-H2_sma3nz_stddevNorm",
"logRelF0-H1-A3_sma3nz_amean",
"logRelF0-H1-A3_sma3nz_stddevNorm",
"F1frequency_sma3nz_amean",
"F1frequency_sma3nz_stddevNorm",
"F1bandwidth_sma3nz_amean",
"F1bandwidth_sma3nz_stddevNorm",
"F1amplitudeLogRelF0_sma3nz_amean",
"F1amplitudeLogRelF0_sma3nz_stddevNorm",
"F2frequency_sma3nz_amean",
"F2frequency_sma3nz_stddevNorm",
"F2bandwidth_sma3nz_amean",
"F2bandwidth_sma3nz_stddevNorm",
"F2amplitudeLogRelF0_sma3nz_amean",
"F2amplitudeLogRelF0_sma3nz_stddevNorm",
"F3frequency_sma3nz_amean",
"F3frequency_sma3nz_stddevNorm",
"F3bandwidth_sma3nz_amean",
"F3bandwidth_sma3nz_stddevNorm",
"F3amplitudeLogRelF0_sma3nz_amean",
"F3amplitudeLogRelF0_sma3nz_stddevNorm",
"alphaRatioV_sma3nz_amean",
"alphaRatioV_sma3nz_stddevNorm",
"hammarbergIndexV_sma3nz_amean",
"hammarbergIndexV_sma3nz_stddevNorm",
"slopeV0-500_sma3nz_amean",
"slopeV0-500_sma3nz_stddevNorm",
"slopeV500-1500_sma3nz_amean",
"slopeV500-1500_sma3nz_stddevNorm",
"spectralFluxV_sma3nz_amean",
"spectralFluxV_sma3nz_stddevNorm",
"mfcc1V_sma3nz_amean",
"mfcc1V_sma3nz_stddevNorm",
"mfcc2V_sma3nz_amean",
"mfcc2V_sma3nz_stddevNorm",
"mfcc3V_sma3nz_amean",
"mfcc3V_sma3nz_stddevNorm",
"mfcc4V_sma3nz_amean",
"mfcc4V_sma3nz_stddevNorm",
"alphaRatioUV_sma3nz_amean",
"hammarbergIndexUV_sma3nz_amean",
"slopeUV0-500_sma3nz_amean",
"slopeUV500-1500_sma3nz_amean",
"spectralFluxUV_sma3nz_amean",
"loudnessPeaksPerSec",
"VoicedSegmentsPerSec",
"MeanVoicedSegmentLengthSec",
"StddevVoicedSegmentLengthSec",
"MeanUnvoicedSegmentLength",
"StddevUnvoicedSegmentLength",
"equivalentSoundLevel_dBp"]

def runopensmile(smile,filepath,features):
    data = smile.process_file(filepath)
    #print(data)
    #x = 0
    file_data = []
    file_data.append(features)
    for i in data:
        file_data.append(data[i].tolist()[0])
    return file_data

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def writetofile(num):
    exportname= 'daicwoz'+str(num)+'_fullaudiodata.csv'
    readfromfoldername = "audioparts_"+str(num)
    with open(exportname, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(list_of_features)

        # write multiple rows
        x = readfromfoldername

        for filename in os.listdir(x):
            fn = os.path.join(x,filename)

            if os.path.isfile(fn):
                print(fn)
                #print(runopensmile(smile,fn,x))
                writer.writerow(runopensmile(smile,fn, x))#write row
#================================================================================

#TAKE BIG AUDIO FILE AND CONVERT TO MANY SMALL ONES WITHIN A FOLDER:===========


def cutupaudio(num):
    start_times = []
    end_times = []
    particspeaking = False
    state = False
    prev = [0,0,0,0]

    csvfile = str(num)+"_TRANSCRIPT.csv"
    output_folder = "audioparts_"+str(num)
    audfile = str(num)+"_AUDIO.wav"
    os.makedirs(output_folder)


    df = pd.read_csv(csvfile)

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

#=================================================================================
        
#RUN AREA:
def runwhole(grandnum):
    cutupaudio(grandnum)
    writetofile(grandnum)

runwhole(grand)

#count = 322
# current_directory = os.getcwd()

# all_entries = os.listdir(current_directory)

#     # Filter out only the folder names
# folder_names = [entry for entry in all_entries if os.path.isdir(entry)]

# for folder_name in folder_names:
#     folder_path = os.path.join(current_directory, folder_name)
#     if os.path.isfile(folder_path):
#         print(folder_path)
#         runwhole(count)
#     count += 1