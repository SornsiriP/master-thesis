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
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(self.env.render(mode='rgb_array'))

    def process(frame):
        #print(frame.shape)
        width = 300
        height = 300
        if frame.size == width * height * 3:
            img = np.reshape(frame, [width, height, 3]).astype(
                np.float32)

        else:
            assert False, "Unknown resolution."
        # img = img[:, :, 0] * 0.399 + img[:, :, 1] * 0.587 + \
        #       img[:, :, 2] * 0.114 #to one channel

        #print(img.shape)
        resized_screen = cv2.resize(
            img, (84, 84), interpolation=cv2.INTER_AREA)
        gray_image = cv2.cvtColor(resized_screen, cv2.COLOR_RGB2GRAY)
        ret,thresh = cv2.threshold(gray_image,.59,1,cv2.THRESH_TOZERO)
        # cv2.imshow("camera",np.array(resized_screen, dtype = np.uint8 ))
        # cv2.waitKey(1)

        for i in range(thresh.shape[0]):
            for j in range(thresh.shape[1]):
                if thresh[i][j]>0.59:
                    thresh[i][j] = 1

        y_t = np.moveaxis(resized_screen, 2, 0)
        # print("y_t", y_t.shape)

        # return y_t.astype(np.uint8)
        return thresh

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

