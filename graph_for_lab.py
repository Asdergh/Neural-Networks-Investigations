import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def data_to_graph(file_name):
    with open(file_name, "r") as file:
        data = file.readlines()
        data_list = []
        result_list = []
        plt.style.use("seaborn")
        figure = plt.figure()
        surface = figure.add_subplot()
        for element in data:
            tmp_string = ""
            tmp_data = []
            for simbol in element:
                if (simbol.isdigit() == True):
                    tmp_string += simbol
                else:
                    if len(tmp_string) != 0:
                        tmp_data.append(int(tmp_string))
                    tmp_string = ""

            if len(tmp_data) > 1 :
                data_list.append(tmp_data)
            else:
                data_list.append([tmp_data[0], 0])
        for cores in data_list:
            if len(cores) == 4:
                result_list.append([cores[0], cores[2]])
            elif len(cores) == 2:
                result_list.append([cores[0], 0])
        result_array = np.asarray(result_list).T
        print(result_array.shape)
        surface.plot(range(0, 6008), result_array[0], color="red", alpha=0.3, label="Va", marker="s", markerfacecolor="black", markersize=1)
        surface.plot(range(0, 6008), result_array[1], alpha=0.3, color="blue", label="Vw", marker="o", markerfacecolor="black", markersize=1)
        plt.legend(loc="upper left")
        plt.show()
        
data = pd.read_csv("task_1.txt", delimiter="/t")
print(np.asarray(data))
            
                    
            
        

data_to_graph(file_name="task_1.txt")