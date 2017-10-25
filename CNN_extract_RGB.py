import pandas as pd
import numpy as np
import os
from extractor import Extractor
from tqdm import tqdm

model = Extractor()

maindir = os.getcwd()
dirrr = maindir + '/Resize'


def label(directory):
	maindir = os.getcwd()
	labeldir = maindir + '/Labels'

	directory = os.path.join(labeldir, directory)
	directory += '.csv'
	labels = pd.read_csv(directory)
	labels = labels.query('ChangeOfPace != "None"')
	labels = labels.query('ChangeOfPace != "NONE"')
	labels = labels.query('ChangeOfPace != "NA"')

	labels = labels.dropna()
	
	return labels 



output = maindir + '/FeaturesRGB' 
directories = os.listdir(dirrr)

dframes = []
for directory in directories:
	print(directory)
	indir = os.path.join(dirrr, directory)


	labels = label(directory)
	dframes.append(labels)
	videos = labels['Path']
	for video in tqdm(videos):
		print(video)
		video = video[:-4]
		invideo = os.path.join(indir, video)

		video = video.replace('/', '_')
		outvideo = os.path.join(output, video)

		framename = os.listdir(invideo)
		framename.sort()
		
		seq = []
		for frame in framename:
			inimg = os.path.join(invideo, frame)
			features = model.extract(inimg)
			seq.append(features)

		seq = np.array(seq)
		np.save(outvideo+'.npy', seq)


pd.concat(dframes).to_csv(maindir+'/Labels'+'/Final.csv', index=False)