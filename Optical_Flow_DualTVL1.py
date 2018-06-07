import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

maindir = os.getcwd()
os.mkdir(maindir + '/Data/Flow')
output = maindir + '/Data/Flow'
dirrr = maindir + '/Data/Resize'

framename = os.listdir(dirrr)
framename.sort()

prvs = cv2.imread(os.path.join(dirrr, framename[0]), cv2.IMREAD_GRAYSCALE)
hsv = np.zeros_like(
	cv2.imread(os.path.join(dirrr, framename[0]), cv2.IMREAD_UNCHANGED)
)
hsv[..., 1] = 255

framename = framename[1:]

for frame in tqdm(framename):
	inimg = os.path.join(dirrr, frame)
	outimg = os.path.join(output, frame)

	next = cv2.imread(inimg, cv2.IMREAD_GRAYSCALE)
	flow = np.zeros_like(prvs)
	dualTV = cv2.createOptFlow_DualTVL1()
	flow = dualTV.calc(prvs, next, flow)

	mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
	hsv[..., 0] = ang * 180 / np.pi / 2
	hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
	rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	cv2.imwrite(outimg, rgb)
	prvs = next
