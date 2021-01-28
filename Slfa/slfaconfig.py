import numpy as np
import matplotlib.pyplot as plt
import random
import pandas

population = 50
memplex = 5
variables = 10
upperlimit = 100
lowerlimit = -100
Dmax = 100
total = 0
global_best = 0
n = int(population / memplex)
miteration = 8
fitness = [0 for i in range(population)]
# generating random frogs
frogs = [[0 for i in range(variables)] for j in range(population)]
for i in range(population):
    for j in range(variables):
        frogs[i][j] = (random.random() * upperlimit) + (random.random() * lowerlimit)
        if frogs[i][j] < 0:
            frogs[i][j] = frogs[i][j] + 100


# calculating the fitness value
def fitness_fn(frogs):
    global fitness
    for i in range(population):
        total = 0
        for j in range(variables):
            total = total + frogs[i][j] * frogs[i][j]
        fitness[i] = total


def fit(check):
    total = 0
    for j in range(variables):
        total = total + check[j] * check[j]
    return total


for z in range(population):
    fitness[z] = fit(frogs[z])

# arranging in descending order
for j in range(population - 1):
    for k in range(j + 1, population):
        if fitness[j] < fitness[k]:
            temp = fitness[j]
            fitness[j] = fitness[k]
            fitness[k] = temp
            temp2 = frogs[j]
            frogs[j] = frogs[k]
            frogs[k] = temp2

# assigning the global best
global_best = fitness[0]

# creating the memplexes
pop = 0
memplexes = [[[0 for i in range(variables)] for j in range(n)] for k in range(memplex)]

for i in range(memplex):
    for j in range(n):
        for k in range(variables):
            memplexes[i][j][k] = frogs[pop][k]
        pop += 1

for i in range(memplex):
    print ("memplex", i)
    for j in range(miteration):
        bestfrog = memplexes[i][0]
        worstfrog = memplexes[i][n - 1]
        fw = 0
        fw = fit(worstfrog)
        for t in range(variables):
            di = random.random() * (bestfrog[t] - worstfrog[t])
            if (di < -Dmax):
                di = Dmax
            if (di > Dmax):
                di = Dmax
            worstfrog[t] = worstfrog[t] + di
        fn = 0
        fn = fit(worstfrog)
        print("old worst ", fw, " new worst ", fn)
        if (fn < fw):
            bestfrog = frogs[0]
            for u in range(variables):
                di = random.random() * (bestfrog[u] - worstfrog[u])
                if (di < -Dmax):
                    di = Dmax
                if (di > Dmax):
                    di = Dmax
                worstfrog[u] = worstfrog[u] + di
            fn = fit(worstfrog)
            print("old worst1 ", fw, " new worst1 ", fn)
            if (fn < fw):
                for v in range(variables):
                    worstfrog[v] = random.random() * upperlimit + random.random() * lowerlimit
                memplexes[i][n - 1] = worstfrog
                fn = fit(worstfrog)
                print("old worst2 ", fw, " new worst2 ", fn)