import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from skimage.filters import laplace, sobel, roberts, scharr


def generate(path,blur):
	df = pd.DataFrame({"name":[],"laplacian mean":[],"laplacian var":[],"laplacian max":[],"sobel mean":[],"sobel var":[],"sobel max":[],"roberts mean":[],"roberts var":[],"roberts max":[],"blur":[]})
	for file in os.listdir(path):
		row = {}
		row["name"] = file
		row["blur"] = blur
		img = cv2.imread(os.path.join(path,file),0)
		if img is not None:
			img = cv2.resize(img,(512,512))
			l = laplace(img)
			row["laplacian var"] = l.var()
			row["laplacian mean"] = l.mean()
			row["laplacian max"] = np.max(l)
			l = sobel(img)
			row["sobel var"] = l.var()
			row["sobel mean"] = l.mean()
			row["sobel max"] = np.max(l)
			l = roberts(img)
			row["roberts var"] = l.var()
			row["roberts mean"] = l.mean()
			row["roberts max"] = np.max(l)
			df = df.append(row,ignore_index=True)
			print(file)
	return df

path = "CERTH_ImageBlurDataset/TrainingSet/Artificially-Blurred/"
generate(path,1).to_csv('Artificially-Blurred.csv',index=False)
print("TrainingSet Artificially-Blurred done")

path = "CERTH_ImageBlurDataset/TrainingSet/Naturally-Blurred/"
generate(path,1).to_csv('Naturally-Blurred.csv',index=False)
print("TrainingSet Naturally-Blurred done")

path = "CERTH_ImageBlurDataset/TrainingSet/Undistorted/"
generate(path,0).to_csv('Undistorted.csv',index=False)
print("TrainingSet Undistorted done")

path = "CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet/"
generate(path,-1).drop(['blur'], axis=1).to_csv('DigitalBlurSet.csv',index=False)
print("EvaluationSet DigitalBlurSet done")

path = "CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet/"
generate(path,-1).drop(['blur'], axis=1).to_csv('NaturalBlurSet.csv',index=False)
print("EvaluationSet NaturalBlurSet done")