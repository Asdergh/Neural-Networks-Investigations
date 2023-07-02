import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np
import random as rd
import pandas as pd
import cv2

"""image_numpy = np.zeros(shape=(300, 300))
img = cv2.imread("robot-eset.png")
cv2.imshow("black", image_numpy)
cv2.imshow("robot", img)
cv2.imwrite("new_file.png", img)
if cv2.waitKey(0) == ord("q"):
    cv2.imwrite("new_file.png", img)"""

"""cap = cv2.VideoCapture(0)
iter = 0
while True:
    acces_token, frame = cap.read()
    gray_frames = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray_image", gray_frames)
    cv2.imshow("frames", frame)
    cv2.imwrite(f"png_images/frames_of_picture{iter}.png", frame)
    if cv2.waitKey(1) == ord("q"):
        break
    iter += 1

cap.release()
cv2.destroyAllWindows()"""

"""black_squere = np.zeros(shape=(300, 300, 3)) 225, 123), 1)
cv2.rectangle(black_squere, (100, 50), (200, 300), (123, 123, 12), 1)
cv2.imshow("black", black_squere)
cv2.line(black_squere, (123, 0), (12, 123), (0,
cv2.waitKey(0)"""

def nothin():
    pass

"""cap = cv2.VideoCapture(0)
cv2.namedWindow("image")"""
#cv2.createTrackbar("R", "image", 0, 255, nothin)
#cv2.createTrackbar("G", "image", 0, 255, nothin)
#cv2.createTrackbar("B", "image", 0, 255, nothin)

"""while True:
    acces_token, frame = cap.read()
    #r = cv2.getTrackbarPos("R", "image")
    #g = cv2.getTrackbarPos("G", "image")
    #b = cv2.getTrackbarPos("B", "image")
    #cv2.rectangle(frame, (120, 120), (220, 220), (rd.randint(0, 225), rd.randint(0, 225), rd.randint(0, 225)), 2)
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, np.array([110, 50, 50]), np.array([130, 255, 255]))
    result_image = cv2.bitwise_and(frame, frame, mask = mask)
    new_image = frame[10:140, 13:190]
    cv2.imshow("resize_image", new_image)
    cv2.imshow("frame",frame)
    cv2.imshow("mask", mask)
    cv2.imshow("result_image", result_image)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()"""
image = cv2.imread("png_images/robot-eset.png")
image = cv2.resize(image, (10, 10))
cv2.imshow("robot", image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = np.asarray(image)
cv2.waitKey(0)
print(image)
def depth_map_from_image(file_with_image = None, image = None):
    surface = plt.figure().add_subplot(projection="3d")
    if file_with_image == None:
        xx1_param, xx2_param = np.meshgrid(np.linspace(0, image.shape[0] * image.shape[1], image.shape[0] * image.shape[1]),
                                           np.linspace(0, image.shape[0] * image.shape[1], image.shape[0] * image.shape[1]))
        depth_param = np.zeros(shape=xx1_param.shape)
        print(depth_param.shape)
        for (index_i, row) in enumerate(image):
            for (index_j, item) in enumerate(row):
                #depth_param[index_i][index_j] = image[index_i][index_j][0] ** 2 + image[index_i][index_j][1] ** 2 + image[index_i][index_j][2] ** 2
                depth_param[index_i][index_j] = image[index_i][index_j]
        surface.plot_surface(xx1_param, xx2_param, depth_param, cmap="viridis", alpha=0.4)
    elif image == None:
        image = cv2.imread(file_with_image)
        xx1_param, xx2_param = np.meshgrid(np.linspace(0, image.shape[0] * image.shape[1], image.shape[0] * image.shape[1]),
                                           np.linspace(0, image.shape[0] * image.shape[1], image.shape[0] * image.shape[1]))
        depth_param = np.zeros(shape=xx1_param.shape)
        print(depth_param.shape)
        for (index_i, row) in enumerate(image):
            for (index_j, item) in enumerate(row):
                depth_param[index_i] = image[index_i][index_j][0] ** 2 + image[index_i][index_j][1] ** 2 + image[index_i][index_j][2] ** 2
        surface.plot_surface(xx1_param, xx2_param, depth_param, cmap="viridis", alpha=0.4)
    
    plt.show()
depth_map_from_image(image = image)







