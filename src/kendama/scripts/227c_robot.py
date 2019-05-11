#!/usr/bin/env python

"""
This script uses sensor inputs from the LIDAR and 3D cameras to find objects in a new environment. State updates are determined from the turtlebot IMU and enconders.
Commands to the arm are sent to the turtlebot using the MoveIt Python API (http://docs.ros.org/kinetic/api/moveit_tutorials/html/).

Written by Tony Zheng in Fall 2018 at University of California Berkeley in partnership with Siemens.

Supervised by Professor Francesco, Borrelli
"""
import rospy
import time
import ctypes
import struct
import sys, select, termios, tty
import copy
import tf
#import moveit_commander
#import moveit_msgs.msg
import geometry_msgs.msg
from cvxopt import spmatrix, matrix, solvers
#from cvxopt.solvers import qp


import numpy as np
from numpy import matlib
import random as rand
from scipy.linalg import block_diag
#import sensor_msgs.point_cloud2 as pc2

from numpy import sign
from math import pi,sqrt,atan2,sin,cos
#from moveit_commander.conversions import pose_to_list
from tf.transformations import quaternion_from_euler, euler_from_quaternion


import rosbag
import sys

from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import JointState,Joy,PointCloud2
from std_msgs.msg import Float64,Float64MultiArray,String,Bool
#from laser_geometry import LaserProjection
#from nav_msgs.msg import Odometry,OccupancyGrid
from tf.msg import tfMessage
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import scipy.linalg as sp
from numpy import genfromtxt
import math

import cvxpy as cp 

class RobotController():
    def __init__(self):
        self.N = 8
        self.n = 6
        self.d = 6 
        self.delT = 0.01
        self.x0 = np.zeros((6,1))
        self.last_guess = np.zeros(((self.N+1)*self.n + self.N*self.d,1))

        self.solver = 'primaldual' # current options: 'cvx' or 'intpoint' or 'cvxgeneral' or 'primaldual'

        # which state in the trajectory are we currently closest to?
        self.closestIndex = 0

    def true_velocity_callback(self, messaged):
        # update the robot's joint angles
        self.x0 = messaged.position

    def TrackingControl(self):
        # set parameters
        n = self.n
        d = self.d
        N = self.N
        last_guess = self.last_guess

        # set up the QP
        A,b,F,g,Q,p = self.setUpQP()
        np.savetxt("A.csv", A, delimiter=",")
        np.savetxt("b.csv", b, delimiter=",")
        np.savetxt("F.csv", F, delimiter=",")
        np.savetxt("g.csv", g, delimiter=",")
        np.savetxt("Q.csv", Q, delimiter=",")
        np.savetxt("p.csv", p, delimiter=",")

        inputs_to_apply = np.zeros((1,6))


        p2 = p.reshape(p.size,1)
        g2 = g.reshape(g.size,1)
        b2 = b.reshape(b.size,1)


        if self.solver == 'cvx':
        # METHOD ONE: CVX QP SOLVER
            solvers.options['show_progress'] = False
            'quiet' == True
            res_cons = solvers.qp(2*matrix(Q), matrix(np.transpose(p)), matrix(F), matrix(g), matrix(A), matrix(b))
            if res_cons['status'] == 'optimal':
                feasible = 1

                Solution = np.squeeze(res_cons['x'])

                # are we sending the correct thing?     
                uPred = np.squeeze(np.transpose(np.reshape((Solution[n*(N+1)+np.arange(d)]),(d,1))))        
                inputs_to_apply = np.array([uPred[0], uPred[1], uPred[2], uPred[3], uPred[4], uPred[5]])
            else:
                return
        elif self.solver == 'intpoint':
        # METHOD TWO: BRIAN INTERIOR POINT SOLVER
            Solution = InteriorPoint_PhaseI(Q,p2,F,g2,A,b2,last_guess)
            #self.last_guess = Solution
            uPred = Solution[n*(N+1)+np.arange(d)]
            inputs_to_apply = np.array([uPred[0], uPred[1], uPred[2], uPred[3], uPred[4], uPred[5]])

        elif self.solver == 'cvxgeneral':
            # METHOD THREE: GENERAL CVX SOLVER 
            x = cp.Variable(((N+1)*n + N*d,1))
            objective = cp.Minimize(cp.quad_form(x,Q) + p*x)
            constraints = [cp.matmul(A,x) == b, cp.matmul(F,x)<=g]
            prob = cp.Problem(objective, constraints)

            result = prob.solve()
            uPred = x.value[n*(N+1)+np.arange(d)]
            inputs_to_apply = np.array([uPred[0], uPred[1], uPred[2], uPred[3], uPred[4], uPred[5]])

        elif self.solver == 'primaldual':
            Solution = PrimalDualIP(Q,p2,F,g2,A,b2,last_guess)
            #self.last_guess = Solution
            uPred = Solution[n*(N+1)+np.arange(d)]
            inputs_to_apply = np.array([uPred[0], uPred[1], uPred[2], uPred[3], uPred[4], uPred[5]])


        return inputs_to_apply
        # send the velocity commands over the publisher
        


    def buildIneqMatrices(self):
    # Fx<=g inequality constraints:
    # actuator and state constraints

        n = self.n #state dimension
        d = self.d #input dimension
        N = self.N

        F_constr = np.eye(N*n) # we constrain u0, x1, u1, x2, ... , xN, uN
        F_init = np.zeros((n,n))

        Fx = np.hstack((np.zeros((N*n,n)),F_constr))
        Fu = np.eye(N*d)
        F1 = block_diag(Fx,Fu)
        F = np.vstack((F1,-F1))


        zeros_array = np.zeros((n,1))

        q_max = np.array([[2*np.pi], 
                        [2*np.pi],
                        [2*np.pi],
                        [2*np.pi],
                        [2*np.pi],
                        [2*np.pi]])

        q_min = np.zeros((n,1))

        qx = np.tile(q_max,(N,1))

        qd_max = np.array([[np.pi], 
                        [np.pi],
                        [np.pi],
                        [2*np.pi],
                        [2*np.pi],
                        [2*np.pi]])
        qu = np.tile(qd_max,(N,1))

        g_unit = np.vstack((qx, qu))
        g_unit2 = np.vstack((np.tile(q_min,(N,1)),qu))
        g = np.vstack((g_unit, g_unit))


        return F,g

    def buildEqMatrices(self):
        # for debugging, we have N = 3
        # Ax=b, equality constraints:
        # dynamics
        # initialization

        n = self.n #state dimension
        d = self.d #input dimension
        N = self.N
        delT = self.delT
        x0 = self.x0

        # edit x0 because joints 1 and 3 are flipped during state reading. 
        x0 = np.reshape(x0,(n,1))
        idx = [2,1,0,3,4,5]
        x0 = x0[idx]

        A_unit = np.eye(n)
        B_unit = delT*np.eye(d)

        Gx = np.eye(n * (N + 1))
        Gu = np.zeros((n * (N + 1), d * (N)))

        for i in range(0, N):
            ind1 = n + i * n + np.arange(n)
            ind2x = i * n + np.arange(n)
            ind2u = i * d + np.arange(d)

            Gx[np.ix_(ind1, ind2x)] = -A_unit
            Gu[np.ix_(ind1, ind2u)] = -B_unit

        A = np.hstack((Gx, Gu))

        b = np.zeros((n*(N+1),1))
        b[0:n] = x0

        return A,b 

    def buildCostMatrices(self):
        # for debugging, we have N = 3
        # Ax=b, equality constraints:
        # dynamics
        # initialization

        N = self.N
        delT = self.delT
        qin = self.qin
        n = self.n

        # print 'state to track:', qin

        # qin is an ((N+1)xn)x1 vertical vector stacking the reference commands to track
        qref = np.vstack((np.zeros((n,1)),qin))

        n = 6 #state dimension
        d = 6 #input dimension

        Q1 = 30*np.eye(N*n)
        Q_comp = block_diag(np.eye(n),Q1) #x0 does not have to track the reference
        R = np.eye(N*d)

        Q = block_diag(Q_comp,R)
        p1 = np.vstack((qref, np.zeros((N*d,1))))
        p = -2*np.matmul(np.transpose(p1),Q)


        return Q,p


    def setUpQP(self):
        self.setXRef()
        A,b = self.buildEqMatrices()
        F,g = self.buildIneqMatrices()
        Q,p = self.buildCostMatrices()

        return A,b,F,g,Q,p


    def addTrajectory(self, trajectory):
        self.trackTrajectory = trajectory

    def setXRef(self):
        x0 = self.x0
        N = self.N
        closestIndex = self.closestIndex
        trajtraj = self.trackTrajectory

        # METHOD ONE: use a shifted trajectory
        newIndexRange = closestIndex+1+np.arange(N)
        self.closestIndex = closestIndex + 1

        # build qin
        qin = np.array([])
        idx = [2, 1, 0, 3, 4, 5] # have to account for switched first and third joints
        for row in newIndexRange:
            qin = np.append(qin, trajtraj[row,idx])

        self.qin = np.transpose([qin])



# Non-object functions
def importTrajFromBag(filename):

    bag = rosbag.Bag(filename)

    storedTrajectory=np.zeros((1,6))    
    for topic, msg, t in bag.read_messages(topics=['/joint_states']):
        storedTrajectory = np.append(storedTrajectory, [msg.position],axis=0)
    storedTrajectory = np.delete(storedTrajectory, (0), axis=0)
    trajectory_length = np.shape(storedTrajectory)[0]
    bag.close()

    return storedTrajectory, trajectory_length


def InteriorPoint_PhaseI(Q,q,F,g,A,b,x,eps=1e-6):

    ##################### PHASE I #############################
    ###########################################################
    ndim = A.shape[1]
    #x = np.zeros((ndim,1))#np.random.randn(ndim,1) #initialize x0
    t = 10.0
    mu = 10.0
    Beta = 0.9

    Abar = np.hstack((A,np.zeros((A.shape[0],1)))) 

    s = np.max(F.dot(x)-g)+1 #initialize slack variable

    while(F.shape[0]/t > eps):
        t = mu*t
        while(True):
            if(all(F.dot(x)-g < 0) & np.allclose(A.dot(x),b)):
                break

            gradPhi = ((np.hstack((F,-np.ones((F.shape[0],1))))).T).dot(1/-(F.dot(x)-g-s))
            hessPhi = ((np.hstack((F,-np.ones((F.shape[0],1))))).T).dot(np.diag(np.square(1/-(F.dot(x)-g-s)).flatten())).dot(np.hstack((F,-np.ones((F.shape[0],1)))))
            KKTmatrix = np.vstack((np.hstack((hessPhi,Abar.T)),\
                                   np.hstack((Abar,np.zeros((Abar.shape[0],Abar.shape[0]))))))
            gradF0 = np.vstack((np.zeros(x.shape),1))

            v = np.linalg.lstsq(KKTmatrix,-np.vstack((t*gradF0+gradPhi,A.dot(x)-b)))[0]
            #v = np.linalg.pinv(KKTmatrix).dot(-np.vstack((t*gradF0+gradPhi,Abar.dot(np.vstack((x,s)))-b)))
            
            #check log decrement
            if(v[:ndim+1].T.dot(np.linalg.lstsq(hessPhi,v[:ndim+1])[0])/2 < eps ):
            #if(v[:ndim+1].T.dot(np.linalg.pinv(hessPhi).dot(v[:ndim+1]))/2 < eps ):
                break

            #backtracking linesearch to maintain feasibility
            Alpha = 1
            while(any(F.dot(x+Alpha*v[:ndim])-g-(s+Alpha*v[ndim]) > 0)):
                Alpha = Beta*Alpha

            #check to see if already taking excessively small stepsizes
            if(Alpha <= eps):
                break

            #update x and s
            x = x + Alpha*v[:ndim]
            s = s + Alpha*v[ndim]

            
    ##################### PHASE II ############################
    ###########################################################

    #use x0 returned from Phase I
    t = 1000.0
    mu = 10.0
    Beta = 0.9
    
    while(F.shape[0]/t > eps):
        t = mu*t
        while(True):
            
            gradPhi2 = F.T.dot(1/-(F.dot(x)-g))
            hessPhi2 = F.T.dot(np.diag(np.square(1/-(F.dot(x)-g)).flatten())).dot(F)
            KKTmatrix2 = np.vstack((np.hstack((t*(2*Q)+hessPhi2,A.T)),\
                                   np.hstack((A,np.zeros((A.shape[0],A.shape[0]))))))
            gradF02 = t*(2*Q.dot(x)+q)

            #v = np.linalg.lstsq(KKTmatrix2,-np.vstack((gradF02+gradPhi2,A.dot(x)-b)))[0]
            v = np.linalg.inv(KKTmatrix2).dot(-np.vstack((gradF02+gradPhi2,A.dot(x)-b)))

            #check log decrement
            #if(v[:ndim].T.dot(np.linalg.lstsq(2*Q+hessPhi2,v[:ndim])[0])/2 < eps ):
            if(v[:ndim].T.dot(np.linalg.inv(2*Q+hessPhi2).dot(v[:ndim]))/2 < eps ):
                break

            #backtracking linesearch to maintain feasibility
            Alpha = 1
            while(any(F.dot(x+Alpha*v[:ndim])-g> 0)):
                Alpha = Beta*Alpha

            #check to see if already taking excessively small stepsizes
            if(Alpha <= eps):
                break

            #update x and s
            x = x + Alpha*v[:ndim]

    return x



def PrimalDualIP(Q,q,F,g,A,b,x,eps=1e-10):

    ##################### PHASE I #############################
    ###########################################################
    ndim = A.shape[1]

    # if(x==None):
    #     x = np.random.randn(ndim,1) #initialize x0
        
    t = 100.0
    mu = 20.0
    Beta = 0.8

    Abar = np.hstack((A,np.zeros((A.shape[0],1)))) 

    s = np.max(F.dot(x)-g)+1 #initialize slack variable

    while(F.shape[0]/t > eps):
        t = mu*t
        while(True):
            if(all(F.dot(x)-g < 0) & np.allclose(A.dot(x),b)):
                break

            gradPhi = ((np.hstack((F,-np.ones((F.shape[0],1))))).T).dot(1/-(F.dot(x)-g-s))
            hessPhi = ((np.hstack((F,-np.ones((F.shape[0],1))))).T).dot(np.diag(np.square(1/-(F.dot(x)-g-s)).flatten())).dot(np.hstack((F,-np.ones((F.shape[0],1)))))
            KKTmatrix = np.vstack((np.hstack((hessPhi,Abar.T)),\
                                   np.hstack((Abar,np.zeros((Abar.shape[0],Abar.shape[0]))))))
            gradF0 = np.vstack((np.zeros(x.shape),1))

            v = np.linalg.lstsq(KKTmatrix,-np.vstack((t*gradF0+gradPhi,A.dot(x)-b)))[0]
            #v = np.linalg.pinv(KKTmatrix).dot(-np.vstack((t*gradF0+gradPhi,Abar.dot(np.vstack((x,s)))-b)))
            
            #check log decrement
            if(v[:ndim+1].T.dot(np.linalg.lstsq(hessPhi,v[:ndim+1])[0])/2 < eps ):
            #if(v[:ndim+1].T.dot(np.linalg.pinv(hessPhi).dot(v[:ndim+1]))/2 < eps ):
                break

            #backtracking linesearch to maintain feasibility
            Alpha = 1
            while(any(F.dot(x+Alpha*v[:ndim])-g-(s+Alpha*v[ndim]) > 0)):
                Alpha = Beta*Alpha

            #check to see if already taking excessively small stepsizes
            if(Alpha <= eps):
                break

            #update x and s
            x = x + Alpha*v[:ndim]
            s = s + Alpha*v[ndim]

            
    ##################### PHASE II ############################
    ###########################################################

    #use x0 returned from Phase I
    mu = 20.0

    Lambda = 1e-1*np.ones((F.shape[0],1))
    Nu = np.zeros((A.shape[0],1))

    Alpha = 0.1
    Beta = 0.3

    t = mu*F.shape[0]/(-(F.dot(x)-g).T.dot(Lambda))
    oldRvec = np.vstack(((2*Q.dot(x)+q)+F.T.dot(Lambda)+A.T.dot(Nu),\
                         -np.diag(Lambda.flatten()).dot(F.dot(x)-g)-1/t*np.ones((F.shape[0],1)),\
                         A.dot(x)-b))        
    
    while(True):
        KKTmatrix = np.vstack((np.hstack((2*Q,F.T,A.T)),\
                                    np.hstack((-np.diag(Lambda.flatten()).dot(F),-np.diag((F.dot(x)-g).flatten()),np.zeros((F.shape[0],A.shape[0])))),\
                                np.hstack((A,np.zeros((A.shape[0],F.shape[0])),np.zeros((A.shape[0],A.shape[0]))))))

        v = np.linalg.inv(KKTmatrix).dot(-oldRvec)
        
        ratio = np.divide(-Lambda,v[ndim:ndim+F.shape[0]])
        ratio[ratio<0] = 1
        stepSize = 0.99*np.min(np.vstack((1,np.min(ratio))))

        x_new = x+stepSize*v[:ndim]
        Lambda_new = Lambda+stepSize*v[ndim:ndim+F.shape[0]]
        Nu_new = Nu+stepSize*v[ndim+F.shape[0]:]

        while(any((F.dot(x_new)-g) >0) or \
              np.linalg.norm(np.vstack(((2*Q.dot(x_new)+q)+F.T.dot(Lambda_new)+A.T.dot(Nu_new),\
                         -np.diag(Lambda_new.flatten()).dot(F.dot(x_new)-g)-1/t*np.ones((F.shape[0],1)),\
                                        A.dot(x_new)-b))) >\
              (1-Alpha*stepSize)*np.linalg.norm(oldRvec)):
            stepSize = Beta*stepSize
            #update x,lambda,nu
            x_new = x+stepSize*v[:ndim]
            Lambda_new = Lambda+stepSize*v[ndim:ndim+F.shape[0]]
            Nu_new = Nu+stepSize*v[ndim+F.shape[0]:]
            stepSize
            if(stepSize<eps):
                break

        if(stepSize<eps):
            break
        
        x = x_new
        Lambda = Lambda_new
        Nu = Nu_new

        t = mu*F.shape[0]/(-(F.dot(x)-g).T.dot(Lambda))
        oldRvec = np.vstack(((2*Q.dot(x)+q)+F.T.dot(Lambda)+A.T.dot(Nu),\
                         -np.diag(Lambda.flatten()).dot(F.dot(x)-g)-1/t*np.ones((F.shape[0],1)),\
                         A.dot(x)-b))

    return x
############################################################################################################################
################################################## MAIN SCRIPT #############################################################
############################################################################################################################

# intialize the node

sys.path.append('/home/tony/Desktop')
#print sys.path

rospy.init_node('ur5e_control', anonymous=True)

# initialize the controller. 
# this also starts the subscriber to joint states, which we use to set the xreference
UR5E = RobotController()
joint_sub = rospy.Subscriber('joint_states',JointState, UR5E.true_velocity_callback)
command_pub = rospy.Publisher('joint_group_vel_controller/command',Float64MultiArray,queue_size =1)
timing_pub = rospy.Publisher('solver_time',Float64,queue_size=1)

# this bag has to be in the folder 
# for smooth movements, the recorded trajectory needs to be the same recorded frequency as the ROS will run
storedTrajectory, trajectory_length = importTrajFromBag('/home/tony/Desktop/p_scene2_2019-04-23-17-18-22.bag') #100 Hz
#storedTrajectory, trajectory_length = importTrajFromBag('/home/tony/Desktop/circle_trajectory_noweight_2__2019-05-07-10-46-36.bag') #100 Hz
#storedTrajectory, trajectory_length = importTrajFromBag('/home/ur5/Desktop/simrobotfiles/1-0.5-0.5-_2019-04-22-18-04-10.bag')

UR5E.addTrajectory(storedTrajectory)


# now we loop. 
loop_rate = 100
#rate = rospy.Rate(loop_rate)


joint_vels = Float64MultiArray()


while UR5E.closestIndex + UR5E.N < trajectory_length:
    # solve the QP using chosen matrix

    start_time = time.time()
    input_calc = UR5E.TrackingControl()
    
    calc_time = time.time() - start_time
    print calc_time
    timing_pub.publish(calc_time)
    
    # send input to the controller
    joint_vels.data = input_calc
    command_pub.publish(joint_vels)

joint_vels.data = np.array([0,0,0,0,0,0])
command_pub.publish(joint_vels)


print 'Finished'



# A = genfromtxt('/home/ur5/.ros/A.csv', delimiter=',')
# b = genfromtxt('/home/ur5/.ros/b.csv', delimiter=',')
# Q = genfromtxt('/home/ur5/.ros/Q.csv', delimiter=',')
# q = genfromtxt('/home/ur5/.ros/p.csv', delimiter=',')
# F = genfromtxt('/home/ur5/.ros/F.csv', delimiter=',')
# g = genfromtxt('/home/ur5/.ros/g.csv', delimiter=',')

# q = q.reshape(q.size,1)
# g = g.reshape(g.size,1)
# b = b.reshape(b.size,1)

# print np.shape(A)
# print np.shape(b)
# print np.shape(Q)
# print np.shape(q)
# print np.shape(F)
# print np.shape(g)

# Q[:6,:6] = np.identity(6)
# g[g==0] = 2*math.pi

# start = time.time()
# x = InteriorPoint_PhaseI(Q,q,F,g,A,b)
# end = time.time()

# print('Inequalities')
# print(F.dot(x)-g)
# print('Equality Violation')
# print(A.dot(x)-b)
# print('Obtained x:')
# print(x)
# print x.T.dot(Q).dot(x)+q.T.dot(x)
# print(end-start)
