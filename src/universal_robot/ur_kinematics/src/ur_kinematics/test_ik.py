import numpy as np
import sys
import roslib
roslib.load_manifest("ur_kinematics")
from ur_kinematics.ur_kin_py import forward, inverse
from math import cos,sin,pi,atan2,acos,asin,sqrt
import cmath
from numpy import linalg

def T_transform_i_to_iplus(alpha,a,d,theta):
	T = np.array([[cos(theta),-sin(theta),0,a],
		[sin(theta)*cos(alpha),cos(theta)*cos(alpha),-sin(alpha),-sin(alpha)*d],
		[sin(theta)*sin(alpha),cos(theta)*sin(alpha),cos(alpha),cos(alpha)*d],
		[0,0,0,1]])
	return T

d1 =  0.1625;
a2 = -0.42500;
a3 = -0.3922;
d4 =  0.1333;
d5 =  0.0997;
d6 =  0.0996;

alpha = [0,pi/2,0,0,pi/2,-pi/2]
a = [0,0,a2,a3,0,0]
d = [d1,0,0,d4,d5,d6]

# q= [7.788287561538709e-07, -1.5707877439311524, 9.5367431640625e-07, -1.5708004436888636, 3.337860107421875e-06, -4.116688863575746e-06]
q=[0.0,0.0,0.0,0.0,0.0,0.0]
# q = [2.557436098271637e-05, -1.5707877439311524, 9.5367431640625e-07, -1.5708004436888636, 3.337860107421875e-06, 8.106231689453125e-06]
# q = [1.997242275868551, -2.55211939434194, 0.015810012817382812, 0.3141776758381347, 0.014154434204101562, -2.894498411809103]
q =[2.557436098271637e-05, -1.5707877439311524, 9.5367431640625e-07, -1.5708004436888636, 3.337860107421875e-06, 8.106231689453125e-06]
theta = [q[2],q[1],q[0],q[3],q[4],q[5]]


i=0

Tmat = [0,0,0,0,0,0]
for i in range(0,6):
	Tmat[i] = T_transform_i_to_iplus(alpha[i],a[i],d[i],theta[i])
T = np.matmul(Tmat[0],np.matmul(Tmat[1],np.matmul(Tmat[2],np.matmul(Tmat[3],np.matmul(Tmat[4],Tmat[5])))))

print(np.around(T,4))



q =[0.2,-1.57,0,-1.57,0,0]
theta = [q[2],q[1],q[0],q[3],q[4],q[5]]


i=0

Tmat = [0,0,0,0,0,0]
for i in range(0,6):
	Tmat[i] = T_transform_i_to_iplus(alpha[i],a[i],d[i],theta[i])
T = np.matmul(Tmat[0],np.matmul(Tmat[1],np.matmul(Tmat[2],np.matmul(Tmat[3],np.matmul(Tmat[4],Tmat[5])))))

print(np.around(T,4))


# print(Tmat)
# print(T)
# q = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
# x = np.around(forward(np.array(theta)),3)
# print(x)
