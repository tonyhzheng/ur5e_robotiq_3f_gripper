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
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

import numpy as np
np.set_printoptions(suppress=True)
import random as rand
import sensor_msgs.point_cloud2 as pc2

from numpy import sign,eye, around
from numpy.linalg import norm,inv,det
from math import pi,sqrt,atan2,sin,cos, floor
from moveit_commander.conversions import pose_to_list
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import matplotlib.pyplot as plt

from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import JointState,Joy,PointCloud2
from std_msgs.msg import Float64,Float64MultiArray,String,Bool
from laser_geometry import LaserProjection
from nav_msgs.msg import Odometry,OccupancyGrid
from tf.msg import tfMessage
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class ur5e_control(object):
    def __init__(self):
        # initialize node
        rospy.init_node('ur5e_control', anonymous=True)
         # laser_projector = LaserProjection()
        pointlist = []
        iteration = -1
        old = 0
        counter = 0
        delta=0
        startTime = time.time()
        prevtime = time.time()

        # move_it_turtlebot = MoveGroupPythonIntefaceTutorial()

        # set node rate
        loop_rate   = 500#500
        ts          = 1.0 / loop_rate
        rate        = rospy.Rate(loop_rate)
        t_prev      = time.time()
        keypress    = time.time()

        i = 0
        sigfig = 3

        # DH Parameters
        d1 =  0.1625;
        a2 =  0.42500;
        a3 =  0.3922;
        d4 =  0.1333;
        d5 =  0.0997;
        d6 =  0.0996;

        self.alpha = [0,pi/2,0,0,pi/2,-pi/2]
        self.a = [0,0,-a2,-a3,0,0]
        self.d = [d1,0,0,d4,d5,d6]

        self.theta = np.zeros((6,1))+0.00001
        self.Ro=np.array([[1,0,0],
            [0,0,-1],
            [0,1,0]])

        self.Po = np.array([[-a2-a3],
            [-d4-d6],
            [d1-d5]])
        self.wlist = np.array([ [[0],[0],[1]],
            [[0],[-1],[0]],
            [[0],[-1],[0]],
            [[0],[-1],[0]],
            [[0],[0],[-1]],
            [[0],[-1],[0]],
            ])
        self.Qlist = np.array([ [[0],[0],[d1]],
            [[0],[0],[d1]],
            [[-a2],[0],[d1]],
            [[-a2-a3],[0],[d1]],
            [[-a2-a3],[-d4],[0]],
            [[-a2-a3],[0],[d1-d5]],
            ])
        self.J = np.array([[1],[1],[1],[1],[1],[1]])
        self.jacobian = eye(6)
        self.qv = []
        self.prev_bpos = np.zeros((3,1))
        self.ball_t_start = time.time()
        self.blist = []
        self.vellist = []

        joint_traj = JointTrajectory()
        joint_point = JointTrajectoryPoint()
        joint_vels = Float64MultiArray()
        self.joint_vels = Float64MultiArray()
        self.hand_state = Float64MultiArray()
        self.ballstate = Float64MultiArray()

        self.command_pub = rospy.Publisher('/pos_based_pos_traj_controller/command',JointTrajectory,queue_size =1)
        # command_pub2 = rospy.Publisher('/ur_driver/jointspeed',JointTrajectory,queue_size =1)
        self.command_pub3 = rospy.Publisher('joint_group_vel_controller/command',Float64MultiArray,queue_size =1)
        self.currentvelpub = rospy.Publisher('joint_group_vel_controller/joint_vel_current',Float64MultiArray,queue_size =1)

        self.eevel_des_pub = rospy.Publisher('desired_ee_vel', Float64MultiArray,queue_size=1)
        eevel_des = Float64MultiArray()
        eevel_des.data = np.array([0,0,0,0,0,0])


        self.hand_state_pub = rospy.Publisher('hand_state',Float64MultiArray,queue_size =1)
        subname = rospy.Subscriber('joint_states', JointState, self.callback_function)
        ballstate = rospy.Subscriber('/world_view', Vector3, self.ball_callback_function)

        self.mpc_vel = [0,0]
        rospy.Subscriber('/hand_state_opt', Float64MultiArray, self.mpc_callback_function)
        self.ball_state_pub = rospy.Publisher('ball_state',Float64MultiArray,queue_size =1)
        rospy.on_shutdown(self.shutdown)


        joint_traj.joint_names = ['elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint', 'wrist_1_joint', 'wrist_2_joint',
      'wrist_3_joint']

        joint_point.positions = np.array([-0.9774951934814453, -1.6556574306883753, -1.6917484442340296, -0.17756159723315434, 1.5352659225463867, 2.1276955604553223])
        # joint_point.positions = np.array([0,0,0,0,0,0])
        joint_point.velocities = np.array([1,-1,-1,-1,-1,-1])
        joint_point.accelerations = np.array([1,-1,-1,-1,-1,-1])
        # joint_point.velocities = np.array([-0.19644730172630054, -0.1178443083991266, 0.14913797608558607, 0.41509166700461164, 0.14835661279488965, -0.10175914445443687])
        # joint_point.accelerations = np.array([-1.4413459458952043, -0.8646309451201031, 1.0942345113473044, 3.0455531135039218, 1.0885016007834076, -0.7466131070688594])
        joint_point.time_from_start.secs = 10

        joint_vels.data = np.array([0,0,0,0,0,0])

        ready = 0
        joint_traj.points = [joint_point]

        self.jacob_pseudo_inv = np.zeros(6)
        cart_vel_des = np.array([0,0,0,0,0,0])
        self.body_frame = False

        self.prev_time = time.time()
        self.joint_vel_des = np.array([0,0,0,0,0,0])
        self.home = np.array([0,-70*pi/180,-140*pi/180,-60*pi/180,0,0])
        self.cupup = np.array([0,-70*pi/180,-140*pi/180,-60*pi/180,-90*pi/180,0])
        self.movehome = False

        self.count = 0
        self.Tstart = []
        self.Toffset = np.concatenate((np.concatenate((eye(3),np.array([[0],[-0.037338],[0.135142]])),axis=1),np.array([[0,0,0,1]])),axis=0)
        
        while not rospy.is_shutdown():
            # command_pub.publish(joint_traj)
            f = 2
            w = 2*3.14159*f
            current_time = time.time()-startTime
            if current_time>2:
                ready +=1 
            if ready ==1:
                startTime = time.time()
                print(startTime)
                ready +=1

            elif ready>1:
                
                [x,z] = self.get_vel_from_data(current_time)
                
                if self.body_frame:
                    cart_vel_des = np.array([z,-x,0,0,0,0])
                else:
                    cart_vel_des = np.array([x,0,z,0,0,0])

                self.joint_vel_des = np.matmul(self.jacob_pseudo_inv,cart_vel_des)
                joint_vels.data = self.joint_vel_des
                # joint_vels.data = np.array([0,0,0,sin(current_time),0,0])
                # if np.any(self.joint_vel_des > np.pi):
                #     print 'saturated'


                if current_time>15:
                    self.movehome = True

                if self.movehome:
                    joint_vels.data = ((self.home.transpose()-self.theta)/3)

                print(joint_vels.data)
                # publish the desired joint vels to the robot
                self.command_pub3.publish(joint_vels)

                # publish the desired end-effector velocity
                eevel_des.data = cart_vel_des
                self.eevel_des_pub.publish(eevel_des)

            rate.sleep()
        





    def get_vel_from_data(self,t):
        # n = 160
        # xdot = [0, 0.11097, 0.11772, 0.12131, 0.12363, 0.12532, 0.1266, 0.1276, 0.12833, 0.12887, 0.1293, 0.1295, 0.12958, 0.12951, 0.12928, 0.12886, 0.12824, 0.12735, 0.12612, 0.12422, 0.12078, 0.11237, -0.0059112, -0.27085, -0.29491, -0.2979, -0.29942, -0.30043, -0.30115, -0.3017, -0.30219, -0.3025, -0.30269, -0.30275, -0.30274, -0.30257, -0.30226, -0.30185, -0.30125, -0.3004, -0.29924, -0.29759, -0.29496, -0.2901, -0.25596, -0.070168, 0.11091, 0.20342, 0.22427, 0.23379, 0.23969, 0.2438, 0.24703, 0.24965, 0.25174, 0.25352, 0.25506, 0.25644, 0.25764, 0.25865, 0.25962, 0.26037, 0.26112, 0.26177, 0.26239, 0.26297, 0.26352, 0.26393, 0.26431, 0.26459, 0.26485, 0.26505, 0.2652, 0.26524, 0.26519, 0.26509, 0.26503, 0.26487, 0.26484, 0.26471, 0.26456, 0.26436, 0.2642, 0.264, 0.26367, 0.26333, 0.2628, 0.26231, 0.26179, 0.2612, 0.26056, 0.25983, 0.25905, 0.25822, 0.25731, 0.25633, 0.25525, 0.25408, 0.25281, 0.25144, 0.24993, 0.24826, 0.24641, 0.24438, 0.24205, 0.2395, 0.2366, 0.23321, 0.22926, 0.22447, 0.21851, 0.21074, 0.20018, 0.18173, 0.14009, 0.051771, -0.072878, -0.1851, -0.25184, -0.28691, -0.3089, -0.32501, -0.33749, -0.34744, -0.356, -0.36349, -0.37015, -0.37627, -0.38185, -0.38701, -0.39183, -0.39635, -0.40053, -0.40451, -0.40826, -0.41181, -0.41507, -0.41807, -0.42079, -0.42335, -0.42565, -0.42774, -0.42959, -0.4314, -0.43286, -0.43417, -0.43535, -0.43635, -0.43722, -0.43792, -0.4385, -0.43904, -0.43948, -0.43985, -0.44016, -0.4404, -0.44056, -0.4407, -0.44077, -0.44084, -0.44088]
        # zdot = [0, 0.019283, 0.020189, 0.020667, 0.021055, 0.021372, 0.021656, 0.021925, 0.022275, 0.022655, 0.023013, 0.023478, 0.023979, 0.024527, 0.025119, 0.025811, 0.026648, 0.027679, 0.029031, 0.030857, 0.033772, 0.040886, 0.14228, 0.39866, 0.42363, 0.42657, 0.42797, 0.42884, 0.42942, 0.42975, 0.42997, 0.4301, 0.43005, 0.43003, 0.42989, 0.42957, 0.42921, 0.42872, 0.42812, 0.42736, 0.42639, 0.42507, 0.42322, 0.41992, 0.39837, 0.27882, 0.14986, 0.080999, 0.065209, 0.057725, 0.053225, 0.049979, 0.047549, 0.045591, 0.043896, 0.04245, 0.041195, 0.040115, 0.039121, 0.03821, 0.037422, 0.03663, 0.03595, 0.035299, 0.034759, 0.034268, 0.033792, 0.03335, 0.032929, 0.032546, 0.032166, 0.031773, 0.031404, 0.031086, 0.030792, 0.030584, 0.030391, 0.030201, 0.03, 0.029825, 0.029653, 0.029454, 0.029255, 0.029035, 0.028889, 0.02875, 0.028662, 0.028643, 0.028617, 0.02856, 0.028514, 0.028481, 0.028419, 0.028408, 0.028437, 0.028457, 0.028476, 0.028523, 0.028618, 0.02877, 0.028975, 0.029114, 0.029285, 0.02957, 0.029769, 0.030132, 0.030545, 0.030959, 0.031513, 0.032149, 0.032894, 0.033869, 0.035398, 0.038009, 0.043779, 0.05587, 0.072723, 0.086646, 0.093173, 0.095167, 0.095256, 0.095281, 0.094277, 0.093015, 0.091547, 0.089795, 0.087847, 0.085606, 0.083367, 0.081003, 0.078481, 0.07593, 0.073266, 0.070567, 0.067746, 0.064947, 0.062023, 0.059083, 0.056053, 0.053051, 0.050056, 0.047072, 0.044123, 0.041042, 0.038191, 0.035441, 0.032739, 0.030149, 0.02766, 0.025229, 0.022893, 0.020608, 0.018398, 0.016304, 0.014254, 0.012288, 0.010326, 0.0083325, 0.006257, 0.0040299, 0.0012329]
        # dt = 0.025

        # n = 120
        xdot = [0, 0.16407, 0.21478, 0.24655, 0.26877, 0.28315, 0.29149, 0.29453, 0.29232, 0.28502, 0.27083, 0.25, 0.21798, 0.16286, 0.050426, -0.25476, -0.42766, -0.48893, -0.51925, -0.53752, -0.54901, -0.55473, -0.55597, -0.55109, -0.53991, -0.51778, -0.48136, -0.41396, -0.27101, -0.10154, 0.020439, 0.10448, 0.16409, 0.20977, 0.24716, 0.27897, 0.30612, 0.32994, 0.35057, 0.36849, 0.38395, 0.39704, 0.40838, 0.41744, 0.4251, 0.43089, 0.43505, 0.43767, 0.43877, 0.43827, 0.43635, 0.43304, 0.42874, 0.42345, 0.41673, 0.40878, 0.39971, 0.38948, 0.37807, 0.36552, 0.35198, 0.33752, 0.32208, 0.30563, 0.28811, 0.26952, 0.24988, 0.22919, 0.20744, 0.18467, 0.16091, 0.13613, 0.11036, 0.083637, 0.056007, 0.027594, -0.0014244, -0.030887, -0.060672, -0.090548, -0.12013, -0.14904, -0.17702, -0.20385, -0.22936, -0.25338, -0.27566, -0.29611, -0.31504, -0.3322, -0.34743, -0.36149, -0.37341, -0.38367, -0.39206, -0.39875, -0.40462, -0.4098, -0.41329, -0.41565, -0.41676, -0.41602, -0.41468, -0.41339, -0.41119, -0.4083, -0.40534, -0.40055, -0.39658, -0.39303, -0.38974, -0.3867, -0.38341, -0.3793, -0.37619, -0.37376, -0.37183, -0.37018, -0.36935, -0.36828, -0.36878]

        zdot = [0, 0.062531, 0.065626, 0.062017, 0.057179, 0.052277, 0.047796, 0.044092, 0.042464, 0.0423, 0.044164, 0.049687, 0.061537, 0.088135, 0.15149, 0.37121, 0.53246, 0.58936, 0.61199, 0.62308, 0.626, 0.62424, 0.61781, 0.60694, 0.59105, 0.56704, 0.53451, 0.48266, 0.38269, 0.26784, 0.18926, 0.13987, 0.10836, 0.087165, 0.072542, 0.062194, 0.054693, 0.049769, 0.045987, 0.043236, 0.04116, 0.039864, 0.038692, 0.037894, 0.037344, 0.036895, 0.036602, 0.036439, 0.036354, 0.036371, 0.036448, 0.036647, 0.036954, 0.037272, 0.03766, 0.038214, 0.038858, 0.039569, 0.040375, 0.041298, 0.042341, 0.043519, 0.044847, 0.04632, 0.04795, 0.049762, 0.051802, 0.054085, 0.056482, 0.059128, 0.062012, 0.065075, 0.068336, 0.071786, 0.075383, 0.079057, 0.082729, 0.086333, 0.089799, 0.093054, 0.095993, 0.098513, 0.10049, 0.10175, 0.10244, 0.10285, 0.10329, 0.10332, 0.10169, 0.10021, 0.098208, 0.095661, 0.094662, 0.092082, 0.08939, 0.08711, 0.084975, 0.082155, 0.079772, 0.078695, 0.07661, 0.074377, 0.072844, 0.071561, 0.070732, 0.069772, 0.068799, 0.068168, 0.067887, 0.067089, 0.066765, 0.066065, 0.065635, 0.06467, 0.06345, 0.061464, 0.058806, 0.054552, 0.048399, 0.038941, 0.022482]

        dt = 0.025

        n = len(xdot)
        i = int(t/dt)
        
        if i<n-1:
            xdot_interp = self.interp(xdot,dt,t)
            zdot_interp = self.interp(zdot,dt,t)
        else:
            xdot_interp = 0
            zdot_interp = 0


        return [xdot_interp,zdot_interp]


    def interp(self,vec,dt,t):
        i = int(t/dt)
        val = (vec[i+1]-vec[i])*(t%dt)/dt+vec[i]
        return val


    def T_transform_i_to_iplus(self,alpha,a,d,theta):
        T = np.array([[cos(theta),-sin(theta),0,a],
            [sin(theta)*cos(alpha),cos(theta)*cos(alpha),-sin(alpha),-sin(alpha)*d],
            [sin(theta)*sin(alpha),cos(theta)*sin(alpha),cos(alpha),cos(alpha)*d],
            [0,0,0,1]])
        return T

    def getT(self,theta):
        Tmat = [0,0,0,0,0,0]
        for i in range(0,6):
            Tmat[i] = self.T_transform_i_to_iplus(self.alpha[i],self.a[i],self.d[i],theta[i])
        T = np.matmul(Tmat[0],np.matmul(Tmat[1],np.matmul(Tmat[2],np.matmul(Tmat[3],np.matmul(Tmat[4],Tmat[5])))))
        return T

    def mpc_callback_function(self,data):
        vel = np.array([[data.data[2]],[data.data[3]]])
        self.vellist.append(vel)
        if len(self.vellist)>10:
            self.vellist.pop(0)
        velavg = np.mean(self.vellist,axis=0)
        # print(vel)
        # print(velavg)
        self.mpc_vel = [velavg[0],velavg[1]]

    def ball_callback_function(self,data):
        dt = time.time()-self.ball_t_start 
        self.ball_t_start =time.time()
        ball_x = data.x
        ball_y = data.y
        ball_z = data.z
        # print (self.prev_bpos-np.array([[ball_x],[ball_y],[ball_z]]))
        bpos = np.array([[ball_x],[ball_y],[ball_z]])
        self.blist.append(bpos)
        if len(self.blist)>10:
            self.blist.pop(0)
        bposavg = np.mean(self.blist,axis=0)
        bvel = (bposavg-self.prev_bpos)/dt
        self.prev_bpos = bposavg
        self.ballstate.data = np.concatenate((bposavg,bvel),axis=0)
        self.ball_state_pub.publish(self.ballstate)

    def callback_function(self,data):
        # print(1/(time.time()-self.prev_time))
        self.prev_time = time.time()
        q = data.position
        v = data.velocity
        self.theta = [q[2],q[1],q[0],q[3],q[4],q[5]]
        self.qv = [v[2],v[1],v[0],v[3],v[4],v[5]]
        self.joint_vels.data = self.qv
        self.currentvelpub.publish(self.joint_vels)
        self.jacobian =  self.manipJac(self.Ro,self.Po,self.wlist,self.Qlist,self.J,self.theta,self.body_frame)


        if self.count == 0:
            self.Tstart = self.getT(self.theta)
            self.count+=1
            # print(self.count)
        else:
            Tcurr = self.getT(self.theta)-self.Tstart
            Tcup = np.matmul(self.getT(self.theta),self.Toffset)
            # print(self.getT(self.theta)[0:3,3],Tcup[0:3,3])
            self.hand_state.data = np.concatenate((Tcup[0:3,3],np.matmul(self.manipJac(self.Ro,self.Po,self.wlist,self.Qlist,self.J,self.theta,False),self.qv)),axis=0)
            # print(np.matmul(self.jacobian ,self.qv),np.matmul(self.manipJac(self.Ro,self.Po,self.wlist,self.Qlist,self.J,self.theta,False),self.qv))
            # print()
            self.hand_state_pub.publish(self.hand_state)
            # print(Tcurr[0:3,3])


        jtj = np.matmul(self.jacobian,self.jacobian.transpose())
        if det(jtj)!=0:
            self.jacob_pseudo_inv = np.matmul(self.jacobian.transpose(),inv(jtj))
            # print(around(joint_vel_des,4))
            # print()
        else:
            self.jacob_pseudo_inv = zeros(6)

    def skewSym(self,v):
        M = np.array([[0,-v[2],v[1]],
                      [v[2],0,-v[0]],
                      [-v[1],v[0],0]])
        return M

    def manipJac(self,Ro,Po,w,Q,J,theta,frame):
    #frame = 0 for spatial jacobian, 1 for body jacobian
        Gst0 = np.concatenate((np.concatenate((Ro,Po),axis=1),np.array([[0,0,0,1]])),axis=0)
        eXiTheta = np.eye(4)
        xilist = [1]*len(w)
        T = [1]*len(w)

        for i in range(len(w)):
            if J[i] == 1:
                # print(w[i])
                # print(np.cross(-1*w[i].transpose(),Q[i].transpose()).transpose())
                xi = np.concatenate((np.cross(-1*w[i].transpose(),Q[i].transpose()).transpose(), w[i]),axis=0)
            elif J(i) == 2:
                v = w[i]
                xi = np.concatenate((v, np.array([[0],[0],[0]])),axis=0)

            xilist[i] = xi
            wHat = self.skewSym(xi[3:6])
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

        for i in range(1,len(w)):
            g1_iminus1=eye(4)
            for j in range(0,i):
                g1_iminus1 = np.matmul(g1_iminus1,T[j])
            p = g1_iminus1[0:3,3]
            phat = self.skewSym(p)
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
        # Vs = np.matmul(Jacobian,thetadot)
        # # print(Vs)

        p=Gst[0:3,3]
        phat = self.skewSym(p)
        Ad[0:3,0:3] = Gst[0:3,0:3]
        Ad[0:3,3:6] = np.matmul(phat,Gst[0:3,0:3])
        Ad[3:6,0:3] = np.zeros(3)
        Ad[3:6,3:6] = Gst[0:3,0:3]
        # Vb = np.matmul(inv(Ad),Vs)
        # # print(Vb)

        if frame:
            Jacobian = np.matmul(inv(Ad),Jacobian)

        return Jacobian#[Vs,Vb]


    def shutdown(self):
        emptyves = Float64MultiArray()
        emptyves.data = np.array([0,0,0,0,0,0])
        self.command_pub3.publish(emptyves)
        rospy.loginfo("Shutting Down")
        plt.close()


def ssssssssssssss(command_pub3):
    emptyves = Float64MultiArray()
    emptyves.data = np.array([0,0,0,0,0,0])
    command_pub3.publish(emptyves)

def all_close(goal, actual, tolerance):
  """
  Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
  @param: goal       A list of floats, a Pose or a PoseStamped
  @param: actual     A list of floats, a Pose or a PoseStamped
  @param: tolerance  A float
  @returns: bool
  """
  all_equal = True
  if type(goal) is list:
    for index in range(len(goal)):
      if abs(actual[index] - goal[index]) > tolerance:
        return False

  elif type(goal) is geometry_msgs.msg.PoseStamped:
    return all_close(goal.pose, actual.pose, tolerance)

  elif type(goal) is geometry_msgs.msg.Pose:
    return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

  return True

# MoveGroup class imported from MoveIt! http://docs.ros.org/kinetic/api/moveit_tutorials/html/
class MoveGroupPythonIntefaceTutorial(object):
    """MoveGroupPythonIntefaceTutorial"""
    def __init__(self):
        super(MoveGroupPythonIntefaceTutorial, self).__init__()

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this move_it_turtlebot the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:

        group_name = "manipulator" #Found in the srdf!!!!!!

        move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)

        ## END_SUB_TUTORIAL

        ## BEGIN_SUB_TUTORIAL basic_info
        ##
        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        print "============ Planning frame: %s" % planning_frame

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print "============ End effector link: %s" % eef_link

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print "============ Available Planning Groups:", robot.get_group_names()

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print "============ Printing robot state"
        print robot.get_current_state()

        print "============ Printing end effector position"
        print move_group.get_current_pose()
        print ""

        move_group.set_goal_orientation_tolerance(20*pi/180)

        ## END_SUB_TUTORIAL

        # Misc variables
        self.box_name = ''
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names


    def go_to_joint_state(self,angles):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_to_joint_state
        ##
        ## Planning to a Joint Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^^
        ## The Panda's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_ so the first
        ## thing we want to do is move it to a slightly better configuration.
        # We can get the joint values from the group and adjust some of the values:
        joint_goal = move_group.get_current_joint_values()
        # print joint_goal
        # raw_input()


        joint_goal[0] = angles[0]
        joint_goal[1] = angles[1]
        joint_goal[2] = angles[2]
        joint_goal[3] = angles[3]
        joint_goal[4] = angles[4]

        # joint_goal[0] = pi/2
        # joint_goal[1] = -pi/3
        # joint_goal[2] = -pi/3
        # joint_goal[3] = 0
        # joint_goal[4] = 0
        # joint_goal[5] = pi/3
        # joint_goal[6] = 0

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()

        ## END_SUB_TUTORIAL

        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)


    def go_to_pose_goal(self, pos):
        # pos is [x,y,z] vector
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion for this group to a desired pose for the
        ## end-effector:

        pose_goal = geometry_msgs.msg.Pose()
        if len(pos)>3:
            # move_group.set_pose_target([pos[0],pos[1],pos[2],pos[3],pos[4],pos[5]])
            move_group.set_position_target([pos[0],pos[1],pos[2]])
        else:
            move_group.set_position_target([pos[0],pos[1],pos[2]])
        # move_group.set_position_target(pos)

        ## Now, we call the planner to compute the plan and execute it.
        plan = move_group.go(wait=True)
        # print(plan)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()

        ## END_SUB_TUTORIAL

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose().pose
        return plan#all_close(pose_goal, current_pose, 0.01)


    def plan_cartesian_path(self, pos, scale=1):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_cartesian_path
        ##
        ## Cartesian Paths
        ## ^^^^^^^^^^^^^^^
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through. If executing  interactively in a
        ## Python shell, set scale = 1.0.
        ##
        waypoints = []

        wpose = move_group.get_current_pose().pose
        print("WPOSE")
        print(wpose)

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = pos[0]
        pose_goal.position.y = pos[1]
        pose_goal.position.z = pos[2]+0.1

        pose_goal.orientation.x = -0.02991
        pose_goal.orientation.y = 0.6877
        pose_goal.orientation.z = 0.0315
        pose_goal.orientation.w = 0.724
        print(pose_goal)
        waypoints.append(pose_goal)
        pose_goal.position.z =  pos[2]
        print(pose_goal)
        waypoints.append(pose_goal)

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this move_it_turtlebot.
        (plan, fraction) = move_group.compute_cartesian_path(
                                           waypoints,   # waypoints to follow
                                           0.01,        # eef_step
                                           0.0)         # jump_threshold

        # Note: We are just planning, not asking move_group to actually move the robot yet:
        return plan, fraction

        ## END_SUB_TUTORIAL


    def display_trajectory(self, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        robot = self.robot
        display_trajectory_publisher = self.display_trajectory_publisher

        ## BEGIN_SUB_TUTORIAL display_trajectory
        ##
        ## Displaying a Trajectory
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## You can ask RViz to visualize a plan (aka trajectory) for you. But the
        ## group.plan() method does this automatically so this is not that useful
        ## here (it just displays the same trajectory again):
        ##
        ## A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        ## We populate the trajectory_start with our current robot state to copy over
        ## any AttachedCollisionObjects and add our plan to the trajectory.
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        display_trajectory_publisher.publish(display_trajectory);

        ## END_SUB_TUTORIAL


    def execute_plan(self, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL execute_plan
        ##
        ## Executing a Plan
        ## ^^^^^^^^^^^^^^^^
        ## Use execute if you would like the robot to follow
        ## the plan that has already been computed:
        move_group.execute(plan, wait=True)

        ## **Note:** The robot's current joint state must be within some tolerance of the
        ## first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail
        ## END_SUB_TUTORIAL


    def wait_for_state_update(self, box_is_known=False, box_is_attached=False, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene

        ## BEGIN_SUB_TUTORIAL wait_for_scene_update
        ##
        ## Ensuring Collision Updates Are Receieved
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## If the Python node dies before publishing a collision object update message, the message
        ## could get lost and the box will not appear. To ensure that the updates are
        ## made, we wait until we see the changes reflected in the
        ## ``get_attached_objects()`` and ``get_known_object_names()`` lists.
        ## For the purpose of this move_it_turtlebot, we call this function after adding,
        ## removing, attaching or detaching an object in the planning scene. We then wait
        ## until the updates have been made or ``timeout`` seconds have passed
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
          # Test if the box is in attached objects
          attached_objects = scene.get_attached_objects([box_name])
          is_attached = len(attached_objects.keys()) > 0

          # Test if the box is in the scene.
          # Note that attaching the box will remove it from known_objects
          is_known = box_name in scene.get_known_object_names()

          # Test if we are in the expected state
          if (box_is_attached == is_attached) and (box_is_known == is_known):
            return True

          # Sleep so that we give other threads time on the processor
          rospy.sleep(0.1)
          seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False
        ## END_SUB_TUTORIAL

if __name__ == '__main__':
    try:
        settings = termios.tcgetattr(sys.stdin)
        ur5e_control()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    except rospy.ROSInterruptException:
        pass
