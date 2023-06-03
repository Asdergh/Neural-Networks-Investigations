import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

plt.style.use("seaborn")
figure = plt.figure()
surface = figure.add_subplot(projection="3d")
surface.set_xlabel("cores_label_x")
surface.set_ylabel("cores_label_y")
surface.set_zlabel("cores_label_z")

def trajectory(data_type):
    datacores = pd.read_csv("data.csv")
    if data_type != "grid":
        x_cores, y_cores, z_cores = [], [], []
        def trajectory_visualisation(time):
            surface.clear()
            x_cores.append(datacores.iloc[time, [1]])
            y_cores.append(datacores.iloc[time, [2]])
            z_cores.append(datacores.iloc[time, [3]])
            surface.plot(x_cores, y_cores, z_cores, color="red")
        anim = manimation.FuncAnimation(figure, trajectory_visualisation, interval=99)

    else:

        x_cores = np.asarray(datacores.iloc[:, 0:100])
        y_cores = np.asarray(datacores.iloc[:, 100:200])
        z_cores = np.asarray(datacores.iloc[:, 200:300])
        surface.plot_surface(x_cores, y_cores, z_cores)
        i, j, k = [], [], []
        """def surface_simulation(time):
            surface.clear()
            i = x_cores[:, 0:time]
            j = y_cores[:, 0:time]
            k = z_cores[:, 0:time]
            surface.plot_surface(i, j, k, alpha=0.3)
        anim = manimation.FuncAnimation(figure, surface_simulation, interval=100)"""

    plt.show()

def create_new_datacores(x, y, z, file_name="data.csv"):
    try:
        cores = pd.DataFrame(np.array([x, y, z]).T, columns=["x: ", "y: ", "z: "])
        cores.to_csv(file_name)
    except ValueError:
        print("[grid refactoring]!!!")
        cores = pd.DataFrame(np.hstack((x, y, z)))
        print(cores)
        surface.plot_surface(np.asarray(cores.iloc[:, 0:100]), np.asarray(cores.iloc[:, 100:200]), np.asarray(cores.iloc[:, 200:300]))
        cores.to_csv(file_name)


    

theta, psi = np.meshgrid(np.linspace(-np.pi, np.pi, 100),
                         np.linspace(-np.pi, np.pi, 100))

x_cores, y_cores = np.meshgrid(np.linspace(-np.pi, np.pi, 100),
                               np.linspace(-np.pi, np.pi, 100))
z_cores = np.sin(np.sqrt(x_cores ** 2 +  y_cores ** 2)) / np.sqrt(x_cores ** 2 + y_cores ** 2)

create_new_datacores(x_cores, y_cores, z_cores)
#surface.plot_surface(x_cores, y_cores, z_cores)
#plt.show()
trajectory(data_type="grid")

    
