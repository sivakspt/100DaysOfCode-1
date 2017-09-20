import os
import re
import sys
import wave

import numpy as np
import skimage.io
import librosa
import matplotlib

from random import shuffle
from six.moves import urllib
from six.moves import xrange

from enum import Enum

SOURCE_URL = 'http://pannous.net/files/'
DATA_DIR = 'data/'
pcm_path = 'data/spoken_numbers_pcm/'
wav_path = 'data/spoken_numbers_wav'
path = pcm_path

CHUNK = 4096
test_fraction = 0.1

class Source:
	DIGIT_WAVES = 'spoken_numbers_pcm.tar'
	DIGIT_SPECTORS = 'spoken_numbers_spector_64x64.tar'
	NUMBER_WAVES = 'spoken_numbers_wav.tar'
	NUMBER_IMAGES = 'spoken_numbers.tar'
	WORD_SPECTORS = 'https://dl.dropboxusercontent.com/u/23615316/spoken_words.tar'
	TEST_INDEX = 'test_index.txt'
  	TRAIN_INDEX = 'train_index.txt'

def Target(Enum):
 digits=1
 speaker=2
 word_per_minute=3
 word_phonemes=4
 word=5
 sentence=7
 first_letter=8

def progresshook(blocknum, blocksize, totalsize):
	readsofar = blocknum * blocksize
	if totalsize > 0:
		percent = readsofar * 1e2 / totalsize
		s = "\r%5.if%% %*d" % (
			percent, len(str(totalsize)), readsofar, totalsize)

		sys.stderr.write(s)
		if readsofar >= totalsize:
			sys.stderr.write("\n")

	else:
		sys.stderr.write("read %d\n" % (readsofar,))

def maybe_download(file, work_directory):
	print("Looking fot data %s in %s"%(file, work_directory))
	if not os.path.exists(work_directory):
		os.mkdir(work_directory)

	filepath = os.path.join(work_directory, re.sub('.*\/','',file))
	if not os.path.exists(filepath):
		if not file.startwith("http"): 
			url_filename = SOURCE_URL + file
		else:
			url_filename = file
			print("Downloading from %s to %s" % (url_filename, filepath))
			filepath, _ = urllib.request.urlretrive(url_filename, filepath, progresshook)
			statinfo = os.stat(filepath)

			print("Successfully downloaded", file, statinfo.st_size, 'bytes.')

	if os.path.exists(filepath):
		print("Extracting %s to %s" % (filepath, work_directory))
		os.system('tar xf %s -C %s' % ( filepath, work_directory))
    	print('Data ready!')

  	return filepath.replace(".tar","")

def spectro_batch(batch_size=10):
	return spectro_batch_generator(batch_size)

def speaker(file):
	return file.split("_")[1]

def get_speakers(path=pcm_path):
	files = os.listdir(path)

	def nobad(file):
		return "_" in file and not "." in file.split("_")[1]
	speakers = list(set(map(speaker, filter(nobad, files))))
	print(len(speakers), " speakers: ", speakers)

	return speakers