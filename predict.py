import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import laplace, sobel, roberts
import pickle
import cv2

with open('classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

for file in os.listdir('./prediction images/'):
	img = cv2.imread(os.path.join('./prediction images/',file),0)
	if img is not None:
		img = cv2.resize(img,(512,512))
		s = sobel(img)
		r = roberts(img)
		l = laplace(img)
		x = np.array([[l.var(),np.max(l),s.mean(),s.var(),np.max(s),r.mean(),r.var(),r.max()]])
		p = clf.predict(x)[0];
		if p==1:
			print(file," is blur")
		else:
			print(file," is clear")