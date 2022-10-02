import matplotlib.pyplot as plt
import numpy as np

x = []
y1 = [0,0]
y2 = [0,0]
for i in range(-60, 41):
    x.append(i)
plt.plot(x, y1, label='y1')
plt.plot(x, y2, label='y2')
plt.legend()
plt.show()