import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.env = env
        self.observation_space = gym.spaces.box.Box(
            low=0, high=255, shape=(1,64,64), dtype=np.uint8)
        # self.observation_space = gym.spaces.Box(
            # low=0, high=255, shape=(1,84, 84), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(self.env.render(mode='rgb_array'))
        # return ProcessFrame84.segment(self.env.render(mode='rgb_array'))

    def segment(segment):
        return segment.astype(np.uint8)

    def process(frame):
        width = 100
        height = 100
        if frame.size == width * height * 3:
            img = np.reshape(frame, [width, height, 3]).astype(
                np.float32)

        else:
            assert False, "Unknown resolution."
        # img = img[:, :, 0] * 0.399 + img[:, :, 1] * 0.587 + \
        #       img[:, :, 2] * 0.114 #to one channel
        resized_screen = cv2.resize(
            img, (64, 64), interpolation=cv2.INTER_AREA)
        gray_image = cv2.cvtColor(resized_screen, cv2.COLOR_RGB2GRAY)
        # cv2.imshow("camera",np.array(gray_image, dtype = np.uint8 ))
        # cv2.waitKey(1)
        # print(gray_image[0])
        # img = self.threshold(gray_image)  
        gray_image = np.reshape(gray_image, [64, 64, 1])
        y_t = np.moveaxis(gray_image, 2, 0)
        # print("y_t", y_t.shape)

        # print(segmented.shape)
        return y_t.astype(np.uint8)
        
    
    def threshold(gray_image):
        ret,thresh = cv2.threshold(gray_image,.59,0,cv2.THRESH_TOZERO)
        # cv2.imshow("camera",np.array(resized_screen, dtype = np.uint8 ))
        # cv2.waitKey(1)

        for i in range(thresh.shape[0]):    #convert to binary image
            for j in range(thresh.shape[1]):
                if thresh[i][j]>0.59:
                    thresh[i][j] = 255
        thresh = np.reshape(thresh, [64, 64,1])
        thresh = np.moveaxis(thresh, 2, 0)
        return thresh.astype(np.uint8)

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        #print(np.moveaxis(observation, 2, 0).shape)
        #observation = observation/255 #normalization
        return np.moveaxis(observation, 2, 0)

