import cv2
import numpy as np
import os
from tqdm import tqdm

maindir = os.getcwd()
maindir = os.path.join(maindir, 'Data')

output = maindir + '/Resize' 
dirrr = maindir + '/Extraction'
images = os.listdir(dirrr)

os.mkdir(output)

for image in images:
	inimg = os.path.join(dirrr, image)
	outimg = os.path.join(output, image)

	img = cv2.imread(inimg)
	rsz = cv2.resize(img, (299,299))

	cv2.imwrite(outimg, rsz)
