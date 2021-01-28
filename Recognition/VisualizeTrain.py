import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('xx-small')

plt.title('CNN Training')
# plt.ylim([0.0001,2])
plt.ylabel("Accuracy")
plt.xlabel("Epoch")


trainn_x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
trainn_y = [0.3,0.8,0.90,0.91,0.92,0.92,0.92,0.92,0.92,0.93,0.94,0.95,0.96,0.97,0.98]
Test_y = [0.26,0.76,0.87,0.89,0.90,0.90,0.90,0.89,0.89,0.91,0.91,0.92,0.94,0.95,0.96]
err_y = [0.97,0.7,0.4,0.2,0.18,0.16,0.13,0.10,0.09,0.08,0.07,0.05,0.04,0.03,0.02]
plt.plot(trainn_x,trainn_y,label="Train",color="blue")
plt.plot(trainn_x,Test_y,label="Test",color="green")
plt.plot(trainn_x,err_y,label="Train loss",linestyle='dashed',color="red")

# i = 0
# for i in range(0,len(data)):
#     key = labels[i]
#     value = data[i]
#     plt.bar(key,value,label=key)

# m1_t.plot(kind='bar')

# ax = plt.gca()
# plt.xlim([-width, len(m1_t)-width])
# ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10'))

plt.legend(loc="center left",bbox_to_anchor=(1, 0.5),prop=fontP)
# plt.plot(data);
plt.show()