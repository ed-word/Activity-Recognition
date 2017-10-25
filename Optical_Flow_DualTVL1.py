import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


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
	return labels['Path']



output = maindir + '/Flow' 
directories = os.listdir(dirrr)

directories = ['hoopcam-videos-B']
#directories = ['hoopcam-videos-H']
#directories = ['hoopcam-videos-r']
#directories = ['hoopcam-videos-S']

for directory in directories:
	print(directory)
	indir = os.path.join(dirrr, directory)

	outdir = os.path.join(output, directory)
	os.mkdir(outdir)

	folders = os.listdir(indir)
	for folder in folders:
		f = os.path.join(outdir, folder)
		os.mkdir(f)

	videos = label(directory)
	for video in tqdm(videos):
		print(video)
		video = video[:-4]
		invideo = os.path.join(indir, video)

		outvideo = os.path.join(outdir, video)
		os.mkdir(os.path.join(outdir, video))

		

		framename = os.listdir(invideo)
		framename.sort()
		
		prvs = cv2.imread(os.path.join(invideo,framename[0]),cv2.IMREAD_GRAYSCALE)

		hsv = np.zeros_like( cv2.imread(os.path.join(invideo,framename[0]),cv2.IMREAD_UNCHANGED) )
		hsv[...,1] = 255

		framename = framename[1:]
		for frame in framename:
			inimg = os.path.join(invideo, frame)
			outimg = os.path.join(outvideo, frame)


			next = cv2.imread(inimg,cv2.IMREAD_GRAYSCALE)

			flow = np.zeros_like(prvs)
			dualTV = cv2.createOptFlow_DualTVL1()
			flow = dualTV.calc(prvs,next,flow)

			mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
			hsv[...,0] = ang*180/np.pi/2
			hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
			rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

			cv2.imwrite(outimg,rgb)
			prvs = next