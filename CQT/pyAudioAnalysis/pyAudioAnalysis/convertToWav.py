# USAGE:
# convertToWav <folder path> <sampling rate> <number of channels>
#

import glob, sys, os

def getVideoFilesFromFolder(dirPath):
	types = (dirPath+os.sep+'*.avi', dirPath+os.sep+'*.mkv', dirPath+os.sep+'*.mp4', dirPath+os.sep+'*.mp3', dirPath+os.sep+'*.flac',dirPath+os.sep+'*.mov') # the tuple of file types
	files_grabbed = []
	for files in types:
		files_grabbed.extend(glob.glob(files))
	return files_grabbed

def main(argv):
	if (len(argv)==4):
		files = getVideoFilesFromFolder(argv[1])
		samplingRate = int(argv[2])
		channels = int(argv[3])
		dir = argv[4]
	
		for f in files:
			ffmpegString = 'ffmpeg -i ' + '\"' + f + '\"' + ' -ar ' + str(samplingRate) + ' -ac ' + str(channels) + ' ' + dir + '\"' + '.wav'
			os.system(ffmpegString)

if __name__ == '__main__':
	files = getVideoFilesFromFolder("/Users/csy/AnacondaProjects/pyAudioAnalysis/RMD/vedio/needtag170522")
	samplingRate = int(44100)
	channels = int(2)
	dir = "/Users/csy/AnacondaProjects/pyAudioAnalysis/RMD/audio/needtag170522"
	for f in files:
         ffmpegString = 'ffmpeg -i ' + f + ' -ab 160k -ac 2 -ar 44100 -vn ' + os.path.splitext(f)[0] + '.wav'
         print(ffmpegString)
         os.system(ffmpegString)
	#main(sys.argv)
	#ffmpeg -i C:/test.mp4 -ab 160k -ac 2 -ar 44100 -vn audio.wav
	#python convertToWav.py /Users/csy/AnacondaProjects/pyAudioAnalysis/RMD/vedio/needtag170519  44100 2 /Users/csy/AnacondaProjects/pyAudioAnalysis/RMD/audio/needtag170519
    #ffmpeg -i /Users/csy/AnacondaProjects/pyAudioAnalysis/RMD/vedio/needtag170519/09-43-50-045_22_91.mov -ab 160k -ac 2 -ar 44100 -vn /Users/csy/AnacondaProjects/pyAudioAnalysis/RMD/audio/needtag170519/09-43-50-045_22_91.wav