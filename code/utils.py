import numpy as np

#utility functions
def sigmoid(x):
	return 1/(1+np.exp(-x))

def softmax(x):
	return np.exp(x)/np.sum(np.exp(x))