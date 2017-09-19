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