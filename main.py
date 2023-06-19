import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import math as mt
import random as rd

plt.style.use("seaborn")

class DATA_Vis():

    def __init__(self) -> None:
        self.figure = plt.figure()
        self.plane = self.figure.add_subplot(projection="3d")
    
    def load_data(self, file_name):
        self.file_name = file_name
        self.grid_flag = False
        data = pd.read_json(self.file_name)
        if len(data.to_numpy()[0]) == 3:
            self.x_cores = np.asarray(data.iloc[:, 0])
            self.y_cores = np.asarray(data.iloc[:, 1])
            self.z_cores = np.asarray(data.iloc[:, 2])
        else:
            self.x_cores = np.asarray(data.iloc[:, 0:100])
            self.y_cores = np.asarray(data.iloc[:, 100:200])
            self.z_cores = np.asarray(data.iloc[:, 200:300])
            self.grid_flag = True
    
    def plot_graph_animaiton(self):
        
        def graph_surface(time):
            self.plane.clear()
            self.plane.scatter(self.x_cores[:, 0:time], self.y_cores[:, 0:time], self.z_cores[:, 0:time], c=self.z_cores[:, 0:time], cmap="magma", s=0.089)
            self.plane.plot_surface(self.x_cores[:, 0:time], self.y_cores[:, 0:time], self.z_cores[:,0:time], alpha=0.5, cmap="bone")
        
        def graph_trajectory(time):
            self.plane.clear()
            self.plane.scatter(self.x_cores[0:time], self.y_cores[0:time], self.z_cores[0:time], c=self.z_cores[0:time], alpha=0.5, cmap="twilight", s=1.23)
            self.plane.plot(self.x_cores[0:time], self.y_cores[0:time], self.z_cores[0:time], linestyle="--", color="blue", alpha=0.1)
            self.plane.scatter(self.x_cores[time], self.y_cores[time], self.z_cores[time], alpha=0.2, color="red")
            self.plane.quiver(self.x_cores[time], self.y_cores[time], self.z_cores[time], self.x_cores[time + 1], 0, 0, color="red")
            self.plane.quiver(self.x_cores[time], self.y_cores[time], self.z_cores[time], 0, self.y_cores[time + 1], 0, color="blue")
            self.plane.quiver(self.x_cores[time], self.y_cores[time], self.z_cores[time], 0, 0, self.z_cores[time + 1], color="green")
            
        if self.grid_flag == True:
            animo = manimation.FuncAnimation(self.figure, graph_surface, interval=100)
            plt.show()
        else:
            animo = manimation.FuncAnimation(self.figure, graph_trajectory, interval=100)
            plt.show()


data = DATA_Vis()
data.load_data(file_name="trajectory.json")
data.plot_graph_animaiton()

def lorenc(xyz, s=10, r=28, b=2.668):
    rot_x = np.array([[1, 0, 0],
                     [0, np.cos(b), np.sin(b)],
                     [0, -np.sin(b), np.cos(b)]])
    
    rot_y = np.array([[np.cos(b), 0, np.sin(b)],
                     [0, 1, 0],
                     [-np.sin(b), 0, np.cos(b)]])
    
    rot_z = np.array([[np.cos(b), np.sin(b), 0],
                     [-np.sin(b), np.cos(b), 0],
                     [0, 0, 1]])
    
    x, y, z = xyz
    x_curent = s * (y - x)
    y_curent = r * x - y - x * z
    z_curent = x * y - b * z
    cores = np.array([x_curent, y_curent, z_curent])
    #cores = np.dot(cores.T, rot_x)
    #cores = np.dot(cores, rot_y)
    #cores = np.dot(cores, rot_z)
    return np.array(cores)

dt = 0.01
xyz = np.empty((10000 + 1, 3))
xyz[0] = (0., 1., 1.05)
for iter in range(10000):
    xyz[iter + 1] = xyz[iter] + lorenc(xyz[iter]) * dt
#xyz = xyz.T


cores = pd.DataFrame(xyz)
print(cores.to_numpy().shape)
print(cores.to_numpy())
cores.to_json("trajectory.json")




        
        



