import cv2
import numpy as np
import os
from tqdm import tqdm

maindir = os.getcwd()

output = maindir + '/Resize' 
dirrr = maindir + '/Extraction'
directories = os.listdir(dirrr)

for directory in directories:
	print(directory)
	indir = os.path.join(dirrr, directory)

	outdir = os.path.join(output, directory)
	os.mkdir(outdir)

	folders = os.listdir(indir)
	for folder in folders:
		infolder = os.path.join(indir, folder)

		outfolder = os.path.join(outdir,folder)
		os.mkdir(outfolder)

		videos = os.listdir(infolder)
		print(folder)
		for video in tqdm(videos):
			invideo = os.path.join(infolder, video)

			outvideo = os.path.join(outfolder, video)
			os.mkdir(outvideo)

			images = os.listdir(invideo)
			for image in images:
				inimg = os.path.join(invideo, image)

				outimg = os.path.join(outvideo, image)

				img = cv2.imread(inimg)
				rsz = cv2.resize(img, (299,299))

				cv2.imwrite(outimg, rsz)