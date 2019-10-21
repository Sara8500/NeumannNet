import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools


A = np.matrix('3 2; 2 6')
b = np.array([2, -8])
c = 0.0


def f(x, A, b, c):
    return float(0.5*x.T*A*x-b.T*x+c)


def bowl(A, b, c):
    fig = plt.figure(figsize=(10, 8))
    qf = fig.gca(projection='3d')
    size = 20
    x1 = list(np.linspace(-6, 6, size))
    x2 = list(np.linspace(-6, 6, size))
    x1, x2 = np.meshgrid(x1, x2)
    zs = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = np.matrix([[x1[i, j]], [x2[i, j]]])
            zs[i, j] = f(x, A, b, c)
    qf.plot_surface(x1, x2, zs, rstride=1, cstride=1, linewidth=0)
    plt.show()

    return x1, x2, zs


x1, x2, zs = bowl(A, b, c)


def contour_steps(x1, x2, zs, steps=None):

    fig = plt.figure(figsize=(6, 6))
    cp = plt.contour(x1, x2, zs, 10)
    plt.clabel(cp, inline=1, fontsize=10)
    if steps is not None:
        steps = np.matrix(steps)
        plt.plot(steps[:, 0], steps[:, 1], '-o')
    plt.show()


contour_steps(x1, x2, zs)


'''The Method of Steepest Descent and Gradient Descent'''

x = np.matrix([[-2.0], [-2.0]])
steps = [(-2.0, -2.0)]
i = 0
i_max = 1000
eps = 0.01
r = b - A * x
delta = r.T * r


while i < i_max:
    #  for steepest descent we are using this alpha
    #float(delta / float(r.T * (A*r)))
    alpha = 0.12
    x = x + alpha * r
    steps.append((x[0, 0], x[1, 0]))
    r = b - A * x
    delta = r.T * r
    i += 1


contour_steps(x1, x2, zs, steps)


'''Method of conjugate Directions'''

Around = np.matrix([[1, 0], [0, 1]])
bround = np.matrix([[0], [0]])
cround = 0
x1, x2, zs = bowl(Around, bround, cround)


va = np.matrix([[2], [2]])
vb = np.matrix([[2], [-2]])
contour_steps(x1, x2, zs, [(va[0, 0], va[1, 0]), (0, 0), (vb[0, 0], vb[1, 0])])

print(float(va.T * vb))

'''Method of conjugate Gradients'''
x1, x2, zs = bowl(A, b, c)