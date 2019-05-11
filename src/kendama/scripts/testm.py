
import numpy as np
from numpy import sign,eye
from numpy.linalg import norm,inv
from math import pi,sqrt,atan2,sin,cos
#Ro = reference configuration of the tool frame WRT base frame
#Po = configuration of the manipulator
#w = unit vectors along the axis of the joints
#Q = positions on each joint axis
#J = type of joints, 1 is revolute, 2 is prismatic. ex. [1 1 2] RRP
#theta = list of angles to rotate each joint by

def skewSym(v):
    M = np.array([[0,-v[2],v[1]],
                  [v[2],0,-v[0]],
                  [-v[1],v[0],0]])
    return M

def manipJac(Ro,Po,w,Q,J,theta,thetadot):
# print([1]*len(wlist))
    Gst0 = np.concatenate((np.concatenate((Ro,Po),axis=1),np.array([[0,0,0,1]])),axis=0)
    eXiTheta = np.eye(4)
    xilist = [1]*len(wlist)
    T = [1]*len(wlist)

    for i in range(len(wlist)):
        if J[i] == 1:
            # print(w[i])
            # print(np.cross(-1*w[i].transpose(),Q[i].transpose()).transpose())
            xi = np.concatenate((np.cross(-1*w[i].transpose(),Q[i].transpose()).transpose(), w[i]),axis=0)
        elif J(i) == 2:
            v = w[i]
            xi = np.concatenate((v, np.array([[0],[0],[0]])),axis=0)

        xilist[i] = xi
        wHat = skewSym(xi[3:6])
        # print(norm(w[i]))
        if (norm(w[i])!=0)and(J[i]==1):
            eW = eye(3)+wHat*sin(theta[i])+(1-cos(theta[i]))*np.matmul(wHat,wHat)
            Tp = np.matmul(np.matmul(eye(3)-eW,wHat)+np.matmul(w[i],w[i].transpose())*theta[i],xi[0:3])
            T[i]= np.concatenate((np.concatenate((eW,Tp),axis=1),np.array([[0,0,0,1]])),axis=0)
        else:
            w_norm = norm(w[i])
            eW = eye(3)+wHat/w_norm*sin(theta[i]*w_norm)+(1-cos(theta[i]*w_norm))*np.matmul(wHat,wHat)/w_norm**2
            Tp = xi[0:3]*theta[i]
            T[i]= np.concatenate((np.concatenate((eW,Tp),axis=1),np.array([[0,0,0,1]])),axis=0)
        # R = eW[0:3,0:3]
        # I = np.matmul(R,R.transpose())
        # print(R)
        # print(I)
        eXiTheta = np.matmul(eXiTheta,T[i])

    Gst = np.matmul(eXiTheta,Gst0)
    # print(Gst)

    Jacobian = xilist[0]

    for i in range(1,len(wlist)):
        g1_iminus1=eye(4)
        for j in range(0,i):
            g1_iminus1 = np.matmul(g1_iminus1,T[j])
        p = g1_iminus1[0:3,3]
        phat = skewSym(p)
        Ad = eye(6)
        Ad[0:3,0:3] = g1_iminus1[0:3,0:3]
        # print(phat)
        # print(g1_iminus1[0:3,0:3])
        Ad[0:3,3:6] = np.matmul(phat,g1_iminus1[0:3,0:3])
        Ad[3:6,0:3] = np.zeros(3)
        Ad[3:6,3:6] = g1_iminus1[0:3,0:3]
        # print(g1_iminus1)
        # print(p)
        # print(Ad)
        xiprime = np.matmul(Ad,xilist[i])
        Jacobian = np.concatenate((Jacobian,xiprime),axis=1)
    Vs = np.matmul(Jacobian,thetadot)
    # print(Vs)



    p=Gst[0:3,3]
    phat = skewSym(p)
    Ad[0:3,0:3] = Gst[0:3,0:3]
    Ad[0:3,3:6] = np.matmul(phat,Gst[0:3,0:3])
    Ad[3:6,0:3] = np.zeros(3)
    Ad[3:6,3:6] = Gst[0:3,0:3]
    Vb = np.matmul(inv(Ad),Vs)
    # print(Vb)
    # print(Gst0)
    # print(xilist)
    # print(eXiTheta)
    # print(T)
    # print(Gst)
    # print(Jacobian)
    return [Vs,Vb]


d1=.1625
a2=0.425
a3=0.3922
d4=0.1333
d5=0.997
d6=0.996
theta=np.zeros((6,1))#np.array([[0],[0],[0],[0],[0],[0]])
thetadot=np.array([[0],[0],[0],[0],[0],[1]])

q = [-0.3246145248413086, -1.3840902608684083, -0.12294894853700811, -1.4101179403117676, 0.0001354217529296875, -5.1800404683888246e-05]

# theta = [q[2],q[1],q[0],q[3],q[4],q[5]]


Ro=np.array([[1,0,0],
    [0,0,-1],
    [0,1,0]])

Po = np.array([[-a2-a3],
    [-d4-d6],
    [d1-d5]])
wlist = np.array([ [[0],[0],[1]],
    [[0],[-1],[0]],
    [[0],[-1],[0]],
    [[0],[-1],[0]],
    [[0],[0],[-1]],
    [[0],[-1],[0]],
    ])
Qlist = np.array([ [[0],[0],[d1]],
    [[0],[0],[d1]],
    [[-a2],[0],[d1]],
    [[-a2-a3],[0],[d1]],
    [[-a2-a3],[-d4],[0]],
    [[-a2-a3],[0],[d1-d5]],
    ])
J = np.array([[1],[1],[1],[1],[1],[1]])

vel = manipJac(Ro,Po,wlist,Qlist,J,theta,thetadot)

print(vel)