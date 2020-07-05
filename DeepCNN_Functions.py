''' Helper Functions '''

#from tensorflow.keras.callbacks import LearningRateScheduler # A Keras callback. We’ll pass our learning rate schedule to this class which will be called as a callback at the completion of each epoch to calculate our learning rate
import numpy as np

# class LearningRate():
#     def __init__(self, epochs, initAlpha=0.01, factor=0.5, dropEvery=10):
# 		# store the base initial learning rate, drop factor, and
# 		# epochs to drop every
# 		self.initAlpha = initAlpha
# 		self.factor = factor
# 		self.dropEvery = dropEvery
	
#     def __call__(self, epoch):
# 		# compute the learning rate for the current epoch
# 		exp = np.floor((1 + epoch) / self.dropEvery)
# 		alpha = self.initAlpha * (self.factor ** exp)
# 		# return the learning rate
# 		return float(alpha)

def scheduler(epoch):
    # compute the learning rate for the current epoch
    dropEvery = 10
    initAlpha = 0.01
    factor = 0.5
    exp = np.floor((1 + epoch) / dropEvery)
    alpha = initAlpha * (factor ** exp)
    # return the learning rate
    print('lr =', alpha)
    return float(alpha)

    #TODO : stop the deacay (and the whole training after 20 .... lie in the paper)
