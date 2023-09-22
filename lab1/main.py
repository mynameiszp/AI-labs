import numpy as np
import matplotlib.pyplot as plt


# Triangle membership function
def trimf(x, params):
    a, b, c = params
    if x <= a or x >= c:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b <= x < c:
        return (c - x) / (c - b)


x = np.linspace(0, 20, 100)
y = [trimf(i, [2, 10, 18]) for i in x]
plt.plot(x, y)
plt.grid()
plt.title("Triangle membership function")
plt.xlabel("X")
plt.ylabel("Membership value")
plt.show()
