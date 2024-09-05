# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:22:26 2024

@author: 86150
"""

import numpy as np
import matplotlib.pyplot as plt

e_values = np.random.randn(100)
plt.plot(e_values)
plt.show()

ts_length = 100
e_values = []

for i in range(ts_length):
    e = np.random.randn()
    e_values.append(e)

plt.plot(e_values)
plt.show()


ts_length = 100
ϵ_values = []
i = 0
while i < ts_length:
    e = np.random.randn()
    ϵ_values.append(e)
    i = i + 1
plt.plot(ϵ_values)
plt.show()


r = 0.025         # interest rate
T = 50            # end date
b = np.empty(T+1) # an empty NumPy array, to store all b_t
b[0] = 10         # initial balance

for t in range(T):
    b[t+1] = (1 + r) * b[t]

plt.plot(b, label='bank balance')
plt.legend()
plt.show()


### Exercise 1

import numpy as np
import matplotlib.pyplot as plt
## setup
T = 200
alpha = 0.9
x = np.empty(T+1)
x[0] = 0

i = 0
while i<T:
    e = np.random.randn()
    x[i+1] = alpha*x[i] + e
    i = i +1
plt.plot(x)
plt.show()

### Exercise 2
plt.figure()

T = 200
alpha = 0
x1 = np.empty(T+1)
x1[0] = 0

i = 0
while i<T:
    e = np.random.randn()
    x1[i+1] = alpha*x1[i] + e
    i = i +1
plt.plot(x1)


T = 200
alpha = 0.8
x2 = np.empty(T+1)
x2[0] = 0

i = 0
while i<T:
    e = np.random.randn()
    x2[i+1] = alpha*x2[i] + e
    i = i +1
plt.plot(x2)


T = 200
alpha = 0.98
x3 = np.empty(T+1)
x3[0] = 0

i = 0
while i<T:
    e = np.random.randn()
    x3[i+1] = alpha*x3[i] + e
    i = i +1
plt.plot(x3)

plt.legend()
plt.show()

α_values = [0.0, 0.8, 0.98]
T = 200
x = np.empty(T+1)

for α in α_values:
    x[0] = 0
    for t in range(T):
        x[t+1] = α * x[t] + np.random.randn()
    plt.plot(x, label=f'$\\alpha = {α}$')

plt.legend()
plt.show()

### Exercise 3

import numpy as np
import matplotlib.pyplot as plt
## setup
T = 200
alpha = 0.9
x = np.empty(T+1)
x[0] = 0

i = 0
while i<T:
    e = np.random.randn()
    x[i+1] = alpha*abs(x[i]) + e
    i = i +1
plt.plot(x)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
## setup
T = 200
alpha = 0.9
x = np.empty(T+1)
x[0] = 0

i = 0
while i<T:
    e = np.random.randn()
    if x[i] >= 0:
        x[i+1] = alpha*x[i] + e
    else:
        x[i+1] = -alpha*x[i] + e
    i = i +1
plt.plot(x)
plt.show()


n = 1000000 # sample size for Monte Carlo simulation

count = 0
for i in range(n):

    # drawing random positions on the square
    u, v = np.random.uniform(), np.random.uniform()

    # check whether the point falls within the boundary
    # of the unit circle centred at (0.5,0.5)
    d = np.sqrt((u - 0.5)**2 + (v - 0.5)**2)

    # if it falls within the inscribed circle, 
    # add it to the count
    if d < 0.5:
        count += 1

area_estimate = count / n

print(area_estimate * 4)  # dividing by radius**2