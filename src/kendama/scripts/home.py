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
        loop_rate   = 120#500
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

        joint_traj = JointTrajectory()
        joint_point = JointTrajectoryPoint()
        joint_vels = Float64MultiArray()
        self.joint_vels = Float64MultiArray()

        self.command_pub = rospy.Publisher('/pos_based_pos_traj_controller/command',JointTrajectory,queue_size =1)
        # command_pub2 = rospy.Publisher('/ur_driver/jointspeed',JointTrajectory,queue_size =1)
        self.command_pub3 = rospy.Publisher('joint_group_vel_controller/command',Float64MultiArray,queue_size =1)
        self.desiredvelpub = rospy.Publisher('joint_group_vel_controller/joint_vel',Float64MultiArray,queue_size =1)
        subname = rospy.Subscriber('joint_states', JointState, self.callback_function)
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
        self.body_frame = True

        self.prev_time = time.time()
        self.joint_vel_des = np.array([0,0,0,0,0,0])
        self.home = np.array([0,-70*pi/180,-140*pi/180,-60*pi/180,0,0])
        self.cupup = np.array([0,-70*pi/180,-140*pi/180,-60*pi/180,-90*pi/180,0])
        self.movehome = True

        self.count = 0
        self.Tstart = []

        self.home1 = True
        while not rospy.is_shutdown():
            # command_pub.publish(joint_traj)
            f = 2
            w = 2*3.14159*f
            if self.movehome:
                joint_vels.data = ((self.home.transpose()-self.theta)/3)
            else:
                joint_vels.data = ((self.cupup.transpose()-self.theta)/3)
            # print(joint_vels.data )
            ##### [1,2,3,5]    
            # plt.plot(self.joint_vel_des[1], 'ro',self.qv[1],'bs')
            # plt.axis([0, 6, 0, 20])
            # plt.show()    
            self.command_pub3.publish(joint_vels)

            
            # joint_vels.data = np.array([0,0,0.4*sin(2*current_time), 0.4*sin(2*current_time),0,0.4*sin(2*current_time)])
            # joint_vels.data = np.array([0,0,0,0,0,0])
            # joint_vels.data = np.array([0,0,-1*sin(2*current_time),1*sin(2*current_time),0,0])
            # cart_vel_des = np.matmul(self.jacobian,joint_vels.data.transpose())
            # cart_vel_des = np.array([0.3*sin(2*current_time),0.3*sin(2*current_time+pi/2),0,0,0,0])
            # cart_vel_des = np.array([0,0,0.1*sin(2*current_time),0,0,0])
            # cart_vel_des = np.array([0.1*sin(2*current_time),0,0,0,0,0])
            # print(around(cart_vel_des,4))
            # print(around(joint_vels.data ,4))

            # lagrangemult = 0.0001
            # joint_vel_des = np.matmul(self.jacob_pseudo_inv,cart_vel_des)
            # joint_vels.data = joint_vel_des
                # print("failed")

            # print(current_time,ready)
            ############################################################################
            # self.command_pub3.publish(joint_vels)
            ############################################################################


            # joint_point.positions = np.array([cos(3*current_time), sin(3*current_time)-1, -0.21748375933108388, 1.4684332653952596, -0.2202624218007605, 0.08156436078884344])
            # # joint_point.velocities = np.array([cos(3*current_time), sin(3*current_time)-1, -0.21748375933108388, 1.4684332653952596, -0.2202624218007605, 0.08156436078884344])
            # joint_traj.points = [joint_point]




            # command_pub2.publish(joint_traj)
            # a = raw_input("============ Moving arm to pose 1")
            # s1 = move_it_turtlebot.go_to_pose_goal(np.round([0.4,-0.04,.7,0,pi/2,0],2))
            # print(move_it_turtlebot.move_group.get_current_pose())
            # a = raw_input("============ Moving arm to pose 2")
            # s1 = move_it_turtlebot.go_to_pose_goal(np.round([0.4,-0.04,.6,0,pi/2,0],2))
            # print(move_it_turtlebot.move_group.get_current_pose())
            # a = raw_input("aaaaaa")
            rate.sleep()
        
    def get_vel_from_data(self,t):
        x = [0,0,-0.00081689,-0.011094,-0.031058,-0.056056,-0.084166,-0.11247,-0.14095,-0.16958,-0.1983,-0.22691,-0.25526,-0.28339,-0.31128,-0.334,-0.34711,-0.3505,-0.34411,-0.32792,-0.30192,-0.26609,-0.22042,-0.16489,-0.099497,-0.024248,0.060872,0.15586,0.26072,0.37543,0.49998]
        z = [0,0,-0.0091122,-0.023816,-0.028231,-0.028288,-0.021678,-0.015064,-0.0084312,-0.0016289,0.0056046,0.013232,0.020816,0.02843,0.047926,0.078683,0.11097,0.14516,0.1807,0.21765,0.25603,0.29584,0.33713,0.37992,0.42419,0.47,0.51746,0.57062,0.71056,1.5272,1.7046]
        # xdot = [0,-0.0073296,-0.005299,-0.0028572,-0.00045876,0.0013234,0.0035355,0.0062875,0.00951,0.041407,0.17399,0.29302,0.39001,0.4657,0.51802,0.55593,0.58588,0.6113,0.636,0.66678,0.70777,0.76129,0.83566,0.92935,1.0494,1.1996,1.3882,1.6208,1.8829,1.9983,1.9927]
        # zdot = [0,-0.0048055,-0.0076589,-0.006756,-0.0043011,-0.0030583,-0.0018497,0.00036919,-0.037112,-0.1872,-0.073735,0.022131,0.097649,0.14981,0.18741,0.21692,0.24188,0.26597,0.29596,0.33602,0.38841,0.46135,0.55353,0.67175,0.81987,1.0061,1.2359,1.4957,1.6134,1.9897,1.2486e-16]
        # xdot = [0,-2.1723e-12,1.2991,1.9959,1.9976,1.997,1.9964,1.9953,1.9884,1.7162,1.0068,-0.01568,-0.023274,-0.027631,-0.030441,-0.032294,-0.033514,-0.034286,-0.034736,-0.034957,-0.035017,-0.034967,-0.034848,-0.034689,-0.034515,-0.034343,-0.034186,-0.034053,-0.033951,-0.033883,-0.033851,-0.033855,-0.033893,-0.033962,-0.034058,-0.034176,-0.034311,-0.034458,-0.034608,-0.034757,-0.034895,-0.035017,-0.035114,-0.035178,-0.035203,-0.03518,-0.035102,-0.034961,-0.03475,-0.034461,-0.034088,-0.033622,-0.033059,-0.032391,-0.031611,-0.030714,-0.029695,-0.028547,-0.027265,-0.025846,-0.024284,-0.022575,-0.020716,-0.018703,-0.016534,-0.014207,-0.011722,-0.0090759,-0.0062713,-0.003309,-0.00019225,0.0030751,0.0064869,0.010036,0.013713,0.017506,0.021402,0.025384,0.029435,0.033532,0.037654,0.041775,0.045869,0.049907,0.05386,0.057713,0.061414,0.064959,0.068288,0.071388,0.074263,0.076878,0.079224,0.081324,0.083115,0.08463,0.085859,0.086805,0.087457,0.087815,0.087864]
        # zdot = [0,1.3031,0.90163,0.61564,0.61404,0.61199,0.60929,0.60058,0.3383,-0.34979,-1.3527,-0.037649,-0.039896,-0.042008,-0.043522,-0.044633,-0.045365,-0.045799,-0.046021,-0.046096,-0.046064,-0.045964,-0.045823,-0.045668,-0.045515,-0.045379,-0.04526,-0.045171,-0.045114,-0.045097,-0.04511,-0.045165,-0.04525,-0.04536,-0.045498,-0.045648,-0.045819,-0.045991,-0.046159,-0.046322,-0.046474,-0.0466,-0.0467,-0.046754,-0.046764,-0.04673,-0.046635,-0.046474,-0.046237,-0.045924,-0.045525,-0.045034,-0.044446,-0.043752,-0.04295,-0.042032,-0.040995,-0.039833,-0.038543,-0.037119,-0.035558,-0.033858,-0.032016,-0.030029,-0.027892,-0.025606,-0.023171,-0.020583,-0.017847,-0.014959,-0.011926,-0.008748,-0.0054322,-0.0019829,0.0015935,0.0052824,0.009081,0.012966,0.016941,0.020963,0.025031,0.029118,0.033208,0.037263,0.041228,0.045199,0.048972,0.052755,0.056326,0.05961,0.062779,0.065707,0.068216,0.070805,0.072881,0.074746,0.076271,0.077573,0.078498,0.079192,3.2197e-17]
        
        xdot = []
        zdot = []        
        xdot = [0,0.85785,0.93926,0.88239,0.95377,-0.019527,-0.99999,-0.9648,-0.96472,-0.96404,-0.96345,-0.96322,-0.96285,-0.96232,-0.64332,-0.63927,-0.086456,-0.039311,-0.016953,-0.013922,-0.014648,-0.014083,-0.014877,-0.013746,0.9984,0.9904]
        zdot = [0,4.6257e-06,1.006e-06,7.4609e-06,2.5316e-08,1.6986e-05,0.86692,2,2,2,1.9997,2,1.999,1.9993,1.1111,1.1017,0.10242,0.0327,0.0037917,8.8456e-05,0.00040153,1.4406e-05,2.2508e-05,1.9771e-05,0.064171,0.00069072]
        
        xdot = [0,0.50211,0.50477,0.5064,0.50732,0.50746,0.50748,0.50736,0.50699,0.50625,0.50445,0.5,-0.25454,-0.72786,-0.73029,-0.73102,-0.73135,-0.73142,-0.73181,-0.73189,-0.73202,-0.73194,-0.73175,-0.73175,-0.73169,-0.37064,-0.035082,-0.017699,-0.011409,-0.0088196,-0.0071619,-0.0058897,-0.005001,-0.0043455,-0.0038923,-0.0035165,-0.0032494,-0.0030295,-0.0028567,-0.0027464,-0.0027186,-0.0027124,-0.0026685,-0.0026178,-0.0026073,-0.0025843,-0.0025504,-0.0025405,-0.0025132,-0.0024814,-0.0024846]
        zdot = [0,0.0046697,0.0041436,0.0037265,0.0034441,0.0033997,0.0034058,0.0034939,0.0037274,0.0041977,0.0053464,0.0083402,0.63153,1.1782,1.1808,1.1814,1.1814,1.1814,1.1811,1.1809,1.1799,1.1798,1.179,1.1789,1.1789,0.66436,0.092063,0.05674,0.041619,0.034357,0.028899,0.024047,0.020199,0.017032,0.014616,0.012557,0.01092,0.0094106,0.0081155,0.007168,0.0067309,0.0066561,0.005939,0.0054065,0.0052736,0.0049297,0.0048577,0.0048068,0.0047093,0.0047266,0.004814]
        
        xdot = [0, 0.16407, 0.21478, 0.24655, 0.26877, 0.28315, 0.29149, 0.29453, 0.29232, 0.28502, 0.27083, 0.25, 0.21798, 0.16286, 0.050426, -0.25476, -0.42766, -0.48893, -0.51925, -0.53752, -0.54901, -0.55473, -0.55597, -0.55109, -0.53991, -0.51778, -0.48136, -0.41396, -0.27101, -0.10154, 0.020439, 0.10448, 0.16409, 0.20977, 0.24716, 0.27897, 0.30612, 0.32994, 0.35057, 0.36849, 0.38395, 0.39704, 0.40838, 0.41744, 0.4251, 0.43089, 0.43505, 0.43767, 0.43877, 0.43827, 0.43635, 0.43304, 0.42874, 0.42345, 0.41673, 0.40878, 0.39971, 0.38948, 0.37807, 0.36552, 0.35198, 0.33752, 0.32208, 0.30563, 0.28811, 0.26952, 0.24988, 0.22919, 0.20744, 0.18467, 0.16091, 0.13613, 0.11036, 0.083637, 0.056007, 0.027594, -0.0014244, -0.030887, -0.060672, -0.090548, -0.12013, -0.14904, -0.17702, -0.20385, -0.22936, -0.25338, -0.27566, -0.29611, -0.31504, -0.3322, -0.34743, -0.36149, -0.37341, -0.38367, -0.39206, -0.39875, -0.40462, -0.4098, -0.41329, -0.41565, -0.41676, -0.41602, -0.41468, -0.41339, -0.41119, -0.4083, -0.40534, -0.40055, -0.39658, -0.39303, -0.38974, -0.3867, -0.38341, -0.3793, -0.37619, -0.37376, -0.37183, -0.37018, -0.36935, -0.36828, -0.36878]

        zdot = [0, 0.062531, 0.065626, 0.062017, 0.057179, 0.052277, 0.047796, 0.044092, 0.042464, 0.0423, 0.044164, 0.049687, 0.061537, 0.088135, 0.15149, 0.37121, 0.53246, 0.58936, 0.61199, 0.62308, 0.626, 0.62424, 0.61781, 0.60694, 0.59105, 0.56704, 0.53451, 0.48266, 0.38269, 0.26784, 0.18926, 0.13987, 0.10836, 0.087165, 0.072542, 0.062194, 0.054693, 0.049769, 0.045987, 0.043236, 0.04116, 0.039864, 0.038692, 0.037894, 0.037344, 0.036895, 0.036602, 0.036439, 0.036354, 0.036371, 0.036448, 0.036647, 0.036954, 0.037272, 0.03766, 0.038214, 0.038858, 0.039569, 0.040375, 0.041298, 0.042341, 0.043519, 0.044847, 0.04632, 0.04795, 0.049762, 0.051802, 0.054085, 0.056482, 0.059128, 0.062012, 0.065075, 0.068336, 0.071786, 0.075383, 0.079057, 0.082729, 0.086333, 0.089799, 0.093054, 0.095993, 0.098513, 0.10049, 0.10175, 0.10244, 0.10285, 0.10329, 0.10332, 0.10169, 0.10021, 0.098208, 0.095661, 0.094662, 0.092082, 0.08939, 0.08711, 0.084975, 0.082155, 0.079772, 0.078695, 0.07661, 0.074377, 0.072844, 0.071561, 0.070732, 0.069772, 0.068799, 0.068168, 0.067887, 0.067089, 0.066765, 0.066065, 0.065635, 0.06467, 0.06345, 0.061464, 0.058806, 0.054552, 0.048399, 0.038941, 0.022482]

        dt = 0.025
        # xdot = [0,0.0046838,0.016378,0.035033,0.060439,0.091846,0.12773,0.16555,0.20201,0.23702,0.27057,0.30266,0.33331,0.3625,0.39024,0.41654,0.4414,0.46481,0.48678,0.50732,0.52641,0.54408,0.5603,0.5751,0.58846,0.60039,0.61089,0.61995,0.62758,0.63378,0.63854,0.64187,0.64377,0.64423,0.64326,0.64085,0.63701,0.63173,0.62501,0.61686,0.60727,0.59624,0.58377,0.56987,0.55452,0.53775,0.51953,0.49987,0.47874,0.45616,0.43213,0.40663,0.37968,0.35127,0.3214,0.29006,0.25725,0.22297,0.18721,0.14996,0.11123,0.071006,0.029303,-0.013878,-0.058532,-0.10465,-0.15222,-0.20109,-0.25012,-0.29915,-0.34817,-0.39669,-0.44358,-0.48878,-0.53227,-0.57406,-0.61413,-0.6525,-0.68917,-0.72414,-0.7574,-0.78897,-0.81885,-0.84705,-0.87356,-0.89838,-0.92153,-0.94301,-0.96282,-0.98096,-0.99745,-1.0123,-1.0]
        # zdot = [0,0.016184,0.053141,0.081847,0.11052,0.13961,0.1711,0.19272,0.16755,0.14218,0.1166,0.090821,0.06485,0.038681,0.012308,-0.014273,-0.041066,-0.068074,-0.095299,-0.12274,-0.15041,-0.1783,-0.20641,-0.23475,-0.26332,-0.29212,-0.32116,-0.35042,-0.37993,-0.40966,-0.43964,-0.46986,-0.50031,-0.53101,-0.56195,-0.59313,-0.62456,-0.65623,-0.68816,-0.72034,-0.75278,-0.78548,-0.81843,-0.85163,-0.88509,-0.91881,-0.95278,-0.98701,-1.0215,-1.0562,-1.0913,-1.1265,-1.162,-1.1978,-1.2339,-1.2702,-1.3067,-1.3436,-1.3806,-1.418,-1.4556,-1.4934,-1.5315,-1.5699,-1.6085,-1.6474,-1.6865,-1.7259,-1.7655,-1.8054,-1.8455,-1.8859,-1.9266,-1.9675,-2.0087,-2.0502,-2.0919,-2.1338,-2.1761,-2.2186,-2.2614,-2.3044,-2.3477,-2.3913,-2.4352,-2.4793,-2.5238,-2.5685,-2.6135,-2.6588,-2.7044,-2.7503,-2.7964,-2.8429,-2.8897,-2.9368,-2.9842,-3.032,-3.0801,-3.1285,-3.1773]

        # dt = 0.005
        n = len(xdot)
        i = int(t/dt)
        # print(i,n)
        if i<n-1:
            xdot_interp = self.interp(xdot,dt,t)#(xdot[i+1]-xdot[i])*(t%dt)+xdot[i]
            zdot_interp = self.interp(zdot,dt,t)
        else:
            xdot_interp = 0
            zdot_interp = 0
        # xdot_interp = 0.1   
        # zdot_interp = 0
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

    def callback_function(self,data):
        # print(1/(time.time()-self.prev_time))
        self.prev_time = time.time()
        q = data.position
        v = data.velocity
        self.theta = [q[2],q[1],q[0],q[3],q[4],q[5]]
        self.qv = [v[2],v[1],v[0],v[3],v[4],v[5]]
        self.joint_vels.data = self.qv
        self.desiredvelpub.publish(self.joint_vels)
        if self.count == 0:
            self.Tstart = self.getT(self.theta)
            self.count+=1
            # print(self.count)
        else:
            Tcurr = self.getT(self.theta)-self.Tstart
            # print(Tcurr[0:3,3])
        self.jacobian =  self.manipJac(self.Ro,self.Po,self.wlist,self.Qlist,self.J,self.theta,self.body_frame)
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
            if (norm(w[i])!=0 ) and (J[i]==1):
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
