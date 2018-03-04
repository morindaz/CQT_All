import sys
import numpy
import os
import audioBasicIO
import audioFeatureExtraction
import audioTrainTest as aT
import audioSegmentation as aS
from pydub import AudioSegment
import matplotlib.pyplot as plt
import time
import  xml.dom.minidom
import glob

nExp = 1

def main(argv):
    #filename = "diarizationExample.wav"
    filename = argv[1]
    for i in range(nExp):
        [Fs1, x1] = audioBasicIO.readAudioFile(filename)			
        tcls = aS.speakerDiarization(filename, 2,LDAdim = 0, PLOT = False)
        audio = AudioSegment.from_wav(filename)
        audio_length = len(audio)
        change_point = [[] for i in range(2)]
        for j in range(len(tcls) - 1):
            if tcls[j] != tcls[j + 1]:
                change_point[0].append(j)
                change_point[1].append(int(tcls[j]))
            if j == len(tcls) - 2:
                change_point[1].append(int(tcls[j]))
        for num in range(len(change_point[1])):
            if num == 0:
                seg_audio = audio[:audio_length*change_point[0][0]/len(tcls)]
                seg_audio.export(os.path.join(argv[2],"seg0_speaker{0}.wav".format(change_point[1][num])),format="wav")
            elif num == len(change_point[1]) - 1:
                seg_audio = audio[audio_length*change_point[0][num - 1]/len(tcls):]
                seg_audio.export(os.path.join(argv[2],"seg{0}_speaker{1}.wav".format(num,change_point[1][num])),format="wav")
            else:
                seg_audio = audio[audio_length*change_point[0][num - 1]/len(tcls):audio_length*change_point[0][num]/len(tcls)]
                seg_audio.export(os.path.join(argv[2],"seg{0}_speaker{1}.wav".format(num,change_point[1][num])),format="wav")
                
# def cut_type(start,end,video,root_dir1,root_dir2, risk):
#     start -= 1
#     if start < 0:
#         start = 0
#     name = video + '-' + str(start) + '-' + str(end) + '.mov'
#     newname = str(risk) + '-' + name
#     duration = end - start
#     os.system('ffmpeg -i D:/Project/pyAudioAnalysis/RMD/vedio/' + root_dir1 + '/' + video + '.mov -ss ' + str(start) + ' -t ' + str(duration) + ' /Users/ansir/Desktop/yh_video/old_videos/old_type_result/' + root_dir2 + '/' + newname)
#     return newname

file_type = ['identity_ID_Card_number','identity_phone_number','identity_other','job_superficial'
                ,'job_deep','willingness','use','others_another_debit','others_other']
# question_type = ['ask - identity - ID Card number','ask - identity - phone number','ask - identity - other ','ask - job'
#                 ,'ask - willingness','ask - use','ask - others - another debit ','ask - others - other','ask - renhang',
#                 'answer - identity - ID Card number','answer - identity - phone number','answer - identity - other ',
#                 'answer - job','answer - willingness','answer - use','answer - others - another debit ','answer - others - other',
#                 'answer - renhang']
question_type = ['identity - ID Card number','identity - phone number','identity - other','job - superficial','job - deep','willingness','use','others - another debit','others - other']
def find_place(l1,l2,condition):
    return l2[[i for i, x in enumerate(l1) if x == condition][0]]
    
def getAudioFilesFromFolder(dirPath):
	types = (dirPath+os.sep+'*.wav',dirPath+os.sep+'*.avi') # the tuple of file types
	files_grabbed = []
	for files in types:
		files_grabbed.extend(glob.glob(files))
	return files_grabbed

if __name__ == '__main__':
    filename = "E:\\video\\answer_I\\0-09-54-34-913_333_86-339.799-342.533.wav"
    outputPath = "E:\\pingan\\CQT\\audioSegment"
    main(sys.argv)
    #argv = {"","/home/bresinno/Desktop/0-06-58-45-122_TEST_19-34.6-62.733.wav","/home/bresinno/Desktop"}
    # roots = ["needtag170519","needtag170522"]
    # CnumTrue = 0
    # CnumSum = 0
    # PnumTrue = 0
    # PnumSum = 0
    # for root in roots:
    #         directory = "/Users/csy/AnacondaProjects/pyAudioAnalysis/RMD/vedio/" + root
    #         files = getAudioFilesFromFolder(directory)
    #         for audiofile in files:
    #             temp = os.path.splitext(audiofile)[0]
    #             name = temp.split('/')[-1]
    #             ptsfile = directory + "/labels/" + name +".anvil"
    #             if os.path.exists(ptsfile) != True:
    #                 continue
    #             dom = xml.dom.minidom.parse(ptsfile)
    #             tracks = dom.getElementsByTagName('track')
    #             questions_dom = tracks[0].getElementsByTagName('el')
    #             starts = []
    #             ends = []
    #             for q in questions_dom:
    #                 starts.append(int(float(q.getAttribute('start'))*10))
    #                 ends.append(int(float(q.getAttribute('end'))*10))
    #             [Fs1, x1] = audioBasicIO.readAudioFile(audiofile)
    #             tcls = aS.speakerDiarization(starts, audiofile, 2,LDAdim = 0, PLOT = False)
    #             audio = AudioSegment.from_wav(audiofile)
    #             audio_length = len(audio)
    #             change_point = []
    #             for j in range(len(tcls) - 1):
    #                 if tcls[j] != tcls[j + 1]:
    #                     change_point.append(j)
    #             for x in range(len(starts)):
    #                 CnumSum += 1
    #                 for y in range(len(change_point)):
    #                     if starts[x]-10 <= change_point[y] and starts[x]+10 >= change_point[y]:
    #                         CnumTrue += 1
    #             for a in range(len(change_point)):
    #                 PnumSum += 1
    #                 for b in range(len(starts)):
    #                     if starts[b]+20 <= change_point[a] and ends[b]-20 >= change_point[a]:
    #                         PnumTrue += 1
    #             print(audiofile)
    # Coverage = float(CnumTrue)/float(CnumSum)
    # Purity = 1 - float(PnumTrue)/float(PnumSum)
    # print("Coverage:{0} Purity:{1}".format(Coverage,Purity))
    for i in range(nExp):
       [Fs1, x1] = audioBasicIO.readAudioFile(filename)
       tcls = aS.speakerDiarization(filename, 2,LDAdim = 0, PLOT = False)
       audio = AudioSegment.from_wav(filename)
       audio_length = len(audio)
       change_point = [[] for i in range(2)]
       for j in range(len(tcls) - 1):
           if tcls[j] != tcls[j + 1]:
               change_point[0].append(j)
               change_point[1].append(int(tcls[j]))
           if j == len(tcls) - 2:
               change_point[1].append(int(tcls[j]))
       for num in range(len(change_point[1])):
           if num == 0:
               seg_audio = audio[:audio_length*change_point[0][0]/len(tcls)]
               seg_audio.export(os.path.join(outputPath,"seg0_speaker{0}.wav".format(change_point[1][num])),format="wav")
           elif num == len(change_point[1]) - 1:
               seg_audio = audio[audio_length*change_point[0][num - 1]/len(tcls):]
               seg_audio.export(os.path.join(outputPath,"seg{0}_speaker{1}.wav".format(num,change_point[1][num])),format="wav")
           else:
               seg_audio = audio[audio_length*change_point[0][num - 1]/len(tcls):audio_length*change_point[0][num]/len(tcls)]
               seg_audio.export(os.path.join(outputPath,"seg{0}_speaker{1}.wav".format(num,change_point[1][num])),format="wav")