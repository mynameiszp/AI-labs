import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz


# Triangle membership function
def trimf(x, params):
    a, b, c = params
    if x < a or x > c:
        return 0
    elif a <= x <= b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return (c - x) / (c - b)


x = np.linspace(0, 20, 100)
y = [trimf(i, [2, 10, 18]) for i in x]
plt.plot(x, y)
plt.grid()
plt.title("Triangle membership function")
plt.xlabel("X")
plt.ylabel("Membership value")
plt.show()


# Trapezoid membership function
def trapmf(x, params):
    a, b, c, d = params
    if x < a or x > d:
        return 0
    elif b < x <= c:
        return 1
    elif a <= x < b:
        return (x - a) / (b - a)
    elif c < x <= d:
        return (d - x) / (d - c)


x = np.linspace(0, 20, 100)
y = [trapmf(i, [2, 8, 12, 18]) for i in x]
plt.plot(x, y)
plt.grid()
plt.title("Trapezoid membership function")
plt.xlabel("X")
plt.ylabel("Membership value")
plt.show()

# Gaussian membership function
x = np.linspace(-5, 5, 100)
y = fuzz.gaussmf(x, 0.5, 0.5)
plt.plot(x, y)
plt.grid()
plt.title("Gaussian membership function")
plt.xlabel("X")
plt.ylabel("Membership value")
plt.show()

# Gaussian combination membership function
x = np.linspace(-5, 8, 100)
y = fuzz.gauss2mf(x, 0.5, 1, 0.5, 0.8)
y1 = fuzz.gauss2mf(x, 1, 0.5, 1.5, 0.5)
y2 = fuzz.gauss2mf(x, 0.2, 0.3, 1, 2)
plt.plot(x, y)
plt.plot(x, y1)
plt.plot(x, y2)
plt.grid()
plt.title("Gaussian combination membership function")
plt.xlabel("X")
plt.ylabel("Membership value")
plt.show()

# Generalized bell shape membership function
x = np.linspace(-8, 8, 100)
y = fuzz.gbellmf(x, 2, 10, 1)
plt.plot(x, y)
plt.grid()
plt.title("Generalized bell shape membership function")
plt.xlabel("X")
plt.ylabel("Membership value")
plt.show()

# Sigmoidal membership function
x = np.linspace(-8, 8, 100)
y = fuzz.sigmf(x, 2, -10)
plt.plot(x, y)
plt.grid()
plt.title("Sigmoidal membership function")
plt.xlabel("X")
plt.ylabel("Membership value")
plt.show()

# Additional Sigmoidal membership functions
x = np.linspace(-2, 12, 100)
y1 = fuzz.dsigmf(x, 2, 5, 7, 10)
y2 = fuzz.psigmf(x, 2, 5, 7, 10)
plt.plot(x, y1)
plt.plot(x, y2)
plt.grid()
plt.title("Additional Sigmoidal membership functions")
plt.xlabel("X")
plt.ylabel("Membership value")
plt.show()

# Z-function
x = np.linspace(0, 12, 100)
y = fuzz.zmf(x, 2, 10)
plt.plot(x, y)
plt.grid()
plt.title("Z-function")
plt.xlabel("X")
plt.ylabel("Membership value")
plt.show()

# PI-function
x = np.linspace(0, 12, 100)
y = fuzz.pimf(x, 1, 2, 7, 10)
plt.plot(x, y)
plt.grid()
plt.title("PI-function")
plt.xlabel("X")
plt.ylabel("Membership value")
plt.show()

# S-function
x = np.linspace(0, 12, 100)
y = fuzz.smf(x, 2, 10)
plt.plot(x, y)
plt.grid()
plt.title("S-function")
plt.xlabel("X")
plt.ylabel("Membership value")
plt.show()

# Minimax functions
x = np.linspace(-5, 8, 100)
y1 = fuzz.gaussmf(x, 1, 2)
y2 = fuzz.gaussmf(x, 3, 2)
min_func = np.minimum(y1, y2)
plt.plot(x, y1, linestyle='dashed')
plt.plot(x, y2, linestyle='dashed')
plt.plot(x, min_func)
plt.grid()
plt.title("Minimal function")
plt.xlabel("X")
plt.ylabel("Membership value")
plt.show()

max_func = np.maximum(y1, y2)
plt.plot(x, y1, linestyle='dashed')
plt.plot(x, y2, linestyle='dashed')
plt.plot(x, max_func)
plt.grid()
plt.title("Maximal function")
plt.xlabel("X")
plt.ylabel("Membership value")
plt.show()

# Probable interpretation of the conjunctive operator
x = np.linspace(-5, 8, 100)
y1 = fuzz.gaussmf(x, 1, 2)
y2 = fuzz.gaussmf(x, 3, 2)
conjunction = y1 * y2
plt.plot(x, y1, linestyle='dashed')
plt.plot(x, y2, linestyle='dashed')
plt.plot(x, conjunction)
plt.grid()
plt.title("Probable interpretation of the conjunctive operator")
plt.xlabel("X")
plt.ylabel("Membership value")
plt.show()

# Probable interpretation of the disjunction operator
x = np.linspace(-5, 8, 100)
y1 = fuzz.gaussmf(x, 1, 2)
y2 = fuzz.gaussmf(x, 3, 2)
disjunction = y1 + y2 - y1 * y2
plt.plot(x, y1, linestyle='dashed')
plt.plot(x, y2, linestyle='dashed')
plt.plot(x, disjunction)
plt.grid()
plt.title("Probable interpretation of the disjunction operator")
plt.xlabel("X")
plt.ylabel("Membership value")
plt.show()

# Probable interpretation of the negation
x = np.linspace(-5, 8, 100)
y = fuzz.gaussmf(x, 1, 3)
negation = 1 - y
plt.plot(x, y1, linestyle='dashed')
plt.plot(x, negation)
plt.grid()
plt.title("Probable interpretation of the negation")
plt.xlabel("X")
plt.ylabel("Membership value")
plt.show()