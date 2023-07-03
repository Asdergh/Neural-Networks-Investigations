import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
import cv2

plt.style.use("dark_background")
class MathTools():

    def __init__(self, file_name = None, cmap = None, alpha = 0.3) -> None:
        
        self.file_name = file_name
        if type(cmap) == list:
            self.cmap = ListedColormap(cmap)
        else:
            self.cmap = cmap
        self.figure = plt.figure()
        self.surface = self.figure.add_subplot(projection="3d")
        self.alpha = alpha
    
    def plot_surface(self, cores = None):
        if self.file_name != None:

            cores = pd.read_json(self.file_name)
            cores_x = np.asarray(cores.iloc[:, 0:100])
            cores_y = np.asarray(cores.iloc[:, 100:200])
            cores_z = np.asarray(cores.iloc[:, 200:300])

            self.surface.plot_surface(cores_x, cores_y, cores_z, alpha = self.alpha, camp = self.cmap)
        else:
            self.surface.plot_surface(cores[:, 0:100], cores[:, 100:200], cores[:, 200:300], alpha = self.alpha, cmap = self.cmap)
    
    def depth_map_from_image(self, file_with_image = None, image_cv = None):
        if file_with_image != None:
            image = cv2.imread(file_with_image)
            image = cv2.resize(image, (100, 100))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            xx1_param, xx2_param = np.meshgrid(np.linspace(0, image.shape[0], 100),
                                               np.linspace(0, image.shape[1], 100))
            depth_param = np.zeros(shape=(xx1_param.shape))
            for (index_i, row) in enumerate(image):
                for (index_j, item) in enumerate(row):
                    depth_param[index_i][index_j] = image[index_i][index_j]

            self.surface.plot_surface(xx1_param, xx2_param, depth_param, cmap = self.cmap, alpha = self.alpha)
            self.surface.scatter(xx1_param, xx2_param, depth_param, alpha = self.alpha, s=0.1, c=depth_param)
            self.surface.set_xlabel("x coordinate of px")
            self.surface.set_ylabel("y coordinate of px")
            self.surface.set_zlabel("px value")

            plt.show()
    
    def color_BGR_to_HSV_bit(self, color):
        hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        hsv_upper = np.array([hsv_color[0][0][0], 100, 100])
        hsv_lower = np.array([hsv_color[0][0][0] + 10, 255, 255])
        return hsv_upper, hsv_lower
    
    def color_detection_mode(self, color):
        
        upper_bound, lower_bound = self.color_BGR_to_HSV_bit(color)
        cap = cv2.VideoCapture(0)

        while True:

            acces_token, frame = cap.read()
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
            res = cv2.bitwise_and(frame, frame, mask = mask)
            cv2.imshow("mask", mask)
            cv2.imshow("hsv_results", hsv_frame)
            cv2.imshow("res", res)
            
            if cv2.waitKey(1) == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
obj = MathTools(cmap="magma")
obj.depth_map_from_image("png_images/robot-eset.png")
    



        
        




