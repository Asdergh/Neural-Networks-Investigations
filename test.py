import numpy as np
import pandas as pd


x, y = np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100)
z = np.sin(x ** 2 + y ** 3) + np.sqrt(x ** 3 / y ** 3) * np.exp(x - y)

data = pd.DataFrame(np.array([x, y, z]).T, columns=["x: ", "y: ", "z: "])
with open("data.csv", "w") as file:
    data.to_csv(file)

test_data = pd.read_csv("data.csv")
print(test_data)