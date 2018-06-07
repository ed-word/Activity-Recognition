import pandas as pd
import numpy as np
import os
from extractor import Extractor
from tqdm import tqdm

model = Extractor()

maindir = os.getcwd()
os.mkdir(maindir + '/Data/FeaturesRGB')
output = maindir + '/Data/FeaturesRGB'
dirrr = maindir + '/Data/Resize'

framename = os.listdir(dirrr)
framename.sort()

seq = []
for frame in tqdm(framename):
	inimg = os.path.join(dirrr, frame)
	features = model.extract(inimg)
	seq.append(features)

seq = np.array(seq)
np.save(output + '/features.npy', seq)
