from dataclasses import dataclass
import enum
import math
import random
import collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

@dataclass
class Vector:
    x: float
    y: float
    z: float

    def sub(self, other):
        return Vector(self.x - other.x, self.y-other.y, self.z-other.z)

    def mod(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5

    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z

    def cross(self, other):
        return Vector( self.y*other.z - self.z*other.y,
                       self.z*other.x - self.x*other.z,
                       self.x*other.y - self.y*other.x
        )


    def angle(self,other):
        rn = self.cross(other)
        c = self.dot(other) / (self.mod() * other.mod())
        s = rn.mod() / (self.mod() * other.mod())
        angle = math.atan2(s, c)

        if rn.z < 0:
            angle = 2*math.pi-angle

        return angle
        # cp = self.cross(other).mod() / (self.mod() * other.mod())
        # return math.asin(cp)
        # #
        dp = self.dot(other) / -(other.mod() * self.mod())
        # if other.z < 0:
        #     return -math.acos(dp)
        return math.acos(dp)

def c_angle(u, v, w):
        c = u.dot(w)
        s = u.cross(w).dot(v)
        return math.atan2(-s,-c)+math.pi

@dataclass
class Point(Vector):
    pass


# initial base points don't move
A = Point(0,-1,0)
B = Point(1,0,0)
C = Point(0,1,0)

r3 = 3**0.5
r6 = 6**0.5
r12 = 12**0.5
# remaining points are functions of theta
# D and F are only defined when cos(theta) >= 1/r3.
def DEF(theta):
    ct, st = math.cos(theta), math.sin(theta)
    E = Point(r3 * ct, 0, r3 * st) # that's the easy one
    try:
        delta = r6 * st*math.sqrt(3 - r12*ct - 3*ct*ct)
    except Exception as e:
        print(theta, math.cos(theta), math.cos(theta)>= 1./r3)
        raise e
    X1 = (r12*ct - 2 + delta)/(7 - r12*ct-3*ct*ct)
    Y1 = X1+1
    Z1 = (2 - X1*(r3*ct - 1))/(r3*st)
    D = Point(X1,Y1,Z1)

    X2 = (r12*ct - 2 - delta)/(7 - r12*ct-3*ct*ct)
    Y2 = -X2-1
    Z2 = (2 - X2*(r3*ct - 1))/(r3*st)
    F = Point(X2,Y2,Z2)

    return D, E, F

def in_delta(a,b, delta = 1.e-12):
    return (a-b)**2 < delta**2

def test_perp(theta = 4.):
    D,E,F = DEF(theta)
    AB = B.sub(A)
    BC = C.sub(B)
    CD = D.sub(C)
    DE = E.sub(D)
    EF = F.sub(E)
    FA = A.sub(F)

    for v in AB, BC, CD, DE, EF, FA:
        assert in_delta(v.mod(), math.sqrt(2)), f"|v| should be sqrt(2), got {v.mod()}"

    assert in_delta(AB.dot(BC), 0.), f"should be 0, got {AB.dot(BC)}"
    assert in_delta(BC.dot(CD), 0.), f"should be 0, got {BC.dot(CD)}"
    assert in_delta(CD.dot(DE), 0.), f"should be 0, got {CD.dot(DE)}"
    assert in_delta(DE.dot(EF), 0.), f"should be 0, got {DE.dot(EF)}"
    assert in_delta(EF.dot(FA), 0.), f"should be 0, got {EF.dot(FA)}"
    assert in_delta(FA.dot(AB), 0.), f"should be 0, got {FA.dot(AB)}"

eps = 1.e-6
def find_angles(N=1000):

    t0 = math.acos(1./r3) + eps
    t1 = 2.*math.pi - t0
    dt = (t1 - t0)/(N-1)
    t = t0
    angles = collections.defaultdict(list)
    for _ in range(N):
        D,E,F = DEF(t)

        AB = A.sub(B)
        BC = B.sub(C)
        CD = C.sub(D)
        DE = D.sub(E)
        EF = E.sub(F)
        FA = F.sub(A)

        angles["A"].append(c_angle(AB, BC, CD))
        angles["B"].append(c_angle(BC, CD, DE))
        angles["C"].append(c_angle(CD, DE, EF))
        angles["D"].append(c_angle(DE, EF, FA))
        angles["E"].append(c_angle(EF, FA, AB))
        angles["F"].append(c_angle(FA, AB, BC))
        angles["th"].append(t)
        t += dt
    return angles


def plot_angles():
    angles = find_angles()
    for k, v in angles.items():
        if k != "th":
            plt.plot(angles["th"], v)
            plt.show()

def numpy_points(th):
    D,E,F = DEF(th)
    return np.array( [
        [A.x, A.y, A.z],
        [B.x, B.y, B.z],
        [C.x, C.y, C.z],
        [D.x, D.y, D.z],
        [E.x, E.y, E.z],
        [F.x, F.y, F.z]
    ])

class AngleAnimation:
    def __init__(self):
        self.fig = plt.figure(figsize=(12,8))
        self.ax3d = self.fig.add_subplot(121,projection="3d")

        self.ax1 = self.fig.add_subplot(322)
        self.ax2 = self.fig.add_subplot(324)
        self.ax3 = self.fig.add_subplot(326)

        angles = find_angles()
        self.t = angles["th"]
        self.a = angles["A"]
        self.b = angles["B"]
        self.c = angles["C"]

        self.all_points = [numpy_points(t) for t in self.t]

        self.frame_count = 0

        self.points = numpy_points(self.t[0])
        self.links = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)]

        self.plot_points = []
        for i in range(len(self.points)):
            p = self.points[i,:]
            pt, = self.ax3d.plot( p[0], p[1], p[2], 'o', markersize=8)
            self.plot_points.append(pt)

        self.lines = []
        for i, j in self.links:
            line, = self.ax3d.plot(
                [self.points[i,0], self.points[j,0]],
                [self.points[i,1], self.points[j,1]],
                [self.points[i,2], self.points[j,2]]
            )
            self.lines.append(line)

        self.ax3d.set_xlim([-1.5, 1.5])  # Adjust to your data range
        self.ax3d.set_ylim([-1.5, 1.5])
        self.ax3d.set_zlim([-1.5, 1.5])

        self.line1, = self.ax1.plot(self.t, self.a)
        self.vline1 = self.ax1.axvline(x=self.t[self.frame_count], color="red")

        self.line2, = self.ax2.plot(self.t, self.b)
        self.vline2 = self.ax2.axvline(x=self.t[self.frame_count], color="red")

        self.line3, = self.ax3.plot(self.t, self.c)
        self.vline3 = self.ax3.axvline(x=self.t[self.frame_count], color="red")

    def update(self, frame):
        self.points = self.all_points[frame]

        for i, pt in enumerate(self.plot_points):
            x,y,z = self.points[i,:]
            pt.set_data_3d([x],[y],[z])

        for (i,j), line in zip(self.links, self.lines):
            line.set_data_3d(
                [self.points[i,0], self.points[j,0]],
                [self.points[i,1], self.points[j,1]],
                [self.points[i,2], self.points[j,2]]
            )
        t = self.t[frame]
        self.vline1.set_xdata([t,t])
        self.vline2.set_xdata([t,t])
        self.vline3.set_xdata([t,t])

        self.frame_count += 1

        return self.plot_points + self.lines + [self.vline1, self.vline2, self.vline3]

def ping_pong(n):
    while True:
        yield from range(n)
        yield from range(n-1,0,-1)

if __name__ == "__main__":
    for _ in range(5):
        while math.cos(theta := random.random() * 2 * math.pi) > 1./r3:
            pass
        test_perp(theta)

    anim = AngleAnimation()
    animation = FuncAnimation(anim.fig, anim.update, frames = ping_pong(1000), interval = 10, blit=True)
    plt.show()
