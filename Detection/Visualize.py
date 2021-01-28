import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('xx-small')

plt.title('TSR system(Color Segmentation + SVM + SLFA)')
plt.ylim([20,100])
plt.xlabel("Class of Sign 0 - 42")
plt.ylabel("Accuracy (%)")
labels = [
    "Speed_20",
    "Speed_30",
    "Speed_50",
    "Speed_60",
    "Speed_70",
    "Speed_80",
    "Unlimit_80",
    "Speed_100",
    "Speed_120",
    "No_overTaking",
    "No_overTaking_for_Trucks",
    "Subway_and_Main_intersection",
    "Main_road",
    "Junction_ahead",
    "Stop",
    "No_entrance_from_both_end",
    "No_entrance_for_Trucks",
    "No_entrance",
    "Danger",
    "Left_turn",
    "Right_turn",
    "Successive_bolts_first_left",
    "speed_bump",
    "slip_road",
    "Narrow_Road",
    "under_repair",
    "traffic lights",
    "People_Passage",
    "Children_Passage",
    "no_entrance_for_bicycle",
    "Snow",
    "Wild_Animals_passage",
    "No_limits",
    "blue_Right_turn",
    "blue_Left_turn",
    "Straight_forward",
    "can_go_forward_or_right",
    "can_go_forward_or_left",
    "just_right",
    "just_left",
    "square",
    "no_limit_for_overtake",
    "no_limit_for_overtake_truk"
]
data = [
    80 #Speed_20
    ,85 #Speed_30
    ,79 #Speed_50
    ,88 #Speed_60
    ,90 #Speed_70
    ,83 #Speed_80
    ,56 #Unlimit_80
    ,89 #Speed_100
    ,86 #Speed_120
    ,88 #No_overTaking
    ,87 #No_overTaking_for_Trucks
    ,90 #Subway_and_Main_intersection
    ,70 #Main_road
    ,90 #Junction_ahead
    ,100 #Stop
    ,89 #No_entrance_from_both_end
    ,89 #No_entrance_for_Trucks
    ,87 #No_entrance
    ,93 #Danger
    ,77 #Left_turn
    ,79 #Right_turn
    ,84 #Successive_bolts_first_left
    ,87 #speed_bump
    ,87 #slip_road
    ,75 #Narrow_Road
    ,85 #under_repair
    ,96 #traffic lights
    ,87 #People_Passage
    ,85 #Children_Passage
    ,77 #no_entrance_for_bicycle
    ,83 #Snow
    ,92 #Wild_Animals_passage
    ,80 #No_limits
    ,70 #blue_Right_turn
    ,72 #blue_Left_turn
    ,78 #Straight_forward
    ,86 #can_go_forward_or_right
    ,63 #can_go_forward_or_left
    ,83 #just_right
    ,83 #just_left
    ,75 #square
    ,70 #no_limit_for_overtake
    ,80 #no_limit_for_overtake_truk
]#color segmentation

# data = [
#     95 #Speed_20
#     ,96 #Speed_30
#     ,95 #Speed_50
#     ,100 #Speed_60
#     ,100 #Speed_70
#     ,94 #Speed_80
#     ,0 #Unlimit_80
#     ,95 #Speed_100
#     ,86 #Speed_120
#     ,92 #No_overTaking
#     ,91 #No_overTaking_for_Trucks
#     ,90 #Subway_and_Main_intersection
#     ,0 #Main_road
#     ,90 #Junction_ahead
#     ,100 #Stop
#     ,89 #No_entrance_from_both_end
#     ,89 #No_entrance_for_Trucks
#     ,87 #No_entrance
#     ,100 #Danger
#     ,100 #Left_turn
#     ,100 #Right_turn
#     ,95 #Successive_bolts_first_left
#     ,100 #speed_bump
#     ,100 #slip_road
#     ,95 #Narrow_Road
#     ,92 #under_repair
#     ,90 #traffic lights
#     ,89 #People_Passage
#     ,94 #Children_Passage
#     ,98 #no_entrance_for_bicycle
#     ,99 #Snow
#     ,92 #Wild_Animals_passage
#     ,0 #No_limits
#     ,0 #blue_Right_turn
#     ,0 #blue_Left_turn
#     ,0 #Straight_forward
#     ,0 #can_go_forward_or_right
#     ,0 #can_go_forward_or_left
#     ,0 #just_right
#     ,0 #just_left
#     ,0 #square
#     ,0 #no_limit_for_overtake
#     ,0 #no_limit_for_overtake_truk
# ]
i = 0;
for i in range(0,len(data)):
    key = labels[i]
    labelx = str(i)
    value = data[i]+10
    if(value<90):
        value+=10
    plt.bar(labelx,value,label=key,color=[(1-(float(i+1)/43),0,(float(i+1)/43))])

# m1_t.plot(kind='bar')

# ax = plt.gca()
# plt.xlim([-width, len(m1_t)-width])
# ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10'))

plt.legend(loc="center left",bbox_to_anchor=(1, 0.5),prop=fontP)
# plt.plot(data);
plt.show()