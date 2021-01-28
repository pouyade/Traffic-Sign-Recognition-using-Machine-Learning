import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('xx-small')

plt.title('TSR system Timing')
# plt.ylim([0.0001,2])
plt.xlabel("Algorithm")
plt.ylabel("Time (Ms)")
labels = [
    "Color Seg (Red)",
    "Color Seg (All)",
    "Color Seg + HOG",
    "Color Seg + HOG + SLFA",
]


data = [0.289,0.445,0.760 ,0.524]
accuracy = [0.65,0.83,0.86 ,0.97]

i = 0
for i in range(0,len(data)):
    key = labels[i]
    value = data[i]
    if(i==3):
        plt.bar(key, value, label=key, color="green")
    else:
        plt.bar(key,value,label=key, color="blue")

# m1_t.plot(kind='bar'
#)
# ax = plt.gca()
# plt.xlim([-width, len(m1_t)-width])
# ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10'))

plt.legend(loc="center left",bbox_to_anchor=(1, 0.5),prop=fontP)
# plt.plot(data);
plt.show()