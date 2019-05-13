#!/usr/bin/env python
# convert camera_xyz to world_xyz
import rospy
import time
from geometry_msgs.msg import Vector3
import numpy as np
from numpy import sin, cos 
import scipy
from scipy.optimize import minimize
from math import pi
import cv2
import sys, select, termios, tty
# import msvcrt
# for calibration
settings = termios.tcgetattr(sys.stdin)

sample_c = np.array([[0],[0],[0]])
sample_r = np.array([[0],[0],[0]])

global xc,yc,zc
global xr,yr,zr

xc = 0
yc = 0
zc = 0

xr = 0
yr = 0
zr = 0

def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    # print(key)
    return key

def camera_callback(data):
	global xc,yc,zc
	xc = data.x
	yc = data.y
	zc = data.z
	# print('camera received',xc)

def robot_callback(data):
	global xr,yr,zr
	xr = data.x
	yr = data.y
	zr = data.z

def calibrate():
	# collect 12 pont pair and pass them to calculate p;
	loop_rate = 600
	dt = 1/loop_rate
	rate = rospy.Rate(loop_rate)
	global xc,yc,zc
	global xr,yr,zr
	global sample_c
	global sample_r
	N = 12
	for i in range(0,N):
		# while not rospy.is_shutdown():
			### start to calibrate, press some key to calibrate
			### judge when to start calibration !
		print('When the arm moves to the desired position, press P to continue: ')
		key = 0
		while not rospy.is_shutdown():
			key = getKey()
			# print(key)
		 	if key in ['p', 'P']:
				break

		start = time.time()
		pos_c = np.array([[0.0],[0.0],[0.0]])
		pos_r = np.array([[0.0],[0.0],[0.0]])

		# print('before',pos_c[0])
		k = np.array(0.0)
		while ((time.time() - start < 2) & ~(rospy.is_shutdown())):

			# print('before',(xc+k*pos_c[0][0])/(k+1))
			pos_c[0][0] = (xc+k*pos_c[0][0])/(k+1)
			pos_c[1][0] = (yc+k*pos_c[1][0])/(k+1)
			pos_c[2][0] = (zc+k*pos_c[2][0])/(k+1)
			# print('after',pos_r)

			pos_r[0][0] = (xr+k*pos_r[0][0])/(k+1)
			pos_r[1][0] = (yr+k*pos_r[1][0])/(k+1)
			pos_r[2][0] = (zr+k*pos_r[2][0])/(k+1)

			k = k+1
			rate.sleep() # get the average location of the world point and camera point
		
		print('sample: %d completed' % (i+1))
		sample_c = np.hstack((sample_c,pos_c))
		sample_r = np.hstack((sample_r,pos_r))
		if i==11:
			break;
	# print(sample_c[:,1:N+1])
	# print(sample_c)

	sample_c = np.vstack((sample_c[:,1:N+1], np.ones([1,N])))
	sample_r = np.vstack((sample_r[:,1:N+1], np.ones([1,N])))

	cons = ({'type':'ineq','fun':lambda x: x[0]+pi},
        {'type':'ineq','fun':lambda x: -x[0]+pi},
        {'type':'ineq','fun':lambda x: x[1]+pi},
        {'type':'ineq','fun':lambda x: -x[1]+pi},
        {'type':'ineq','fun':lambda x: x[2]+pi},
        {'type':'ineq','fun':lambda x: -x[2]+pi}
        )

	# print(cost2go(np.array([pi/2,pi/30,pi/8,1,2,3])))
	opt = minimize(fun=cost2go, x0=[0,0,0,0,0,0],constraints = cons)
	# print(cost2go(opt.x))
	print(opt)
	return opt.x

def rotate(ROT, x):
    rotx = np.array(ROT[0])
    roty = np.array(ROT[1])
    rotz = np.array(ROT[2])
    
    transx = np.array(ROT[3])
    transy = np.array(ROT[4])
    transz = np.array(ROT[5])
    
    rotx = np.array([[1,0,0],[0, cos(rotx),-sin(rotx)],[0,sin(rotx),cos(rotx)]])
    rotx = np.array(rotx)
    roty = np.array([[cos(roty),0,sin(roty)],[0,1,0],[-sin(roty),0,cos(roty)]])
    roty = np.array(roty)
    rotz = np.array([[cos(rotz),-sin(rotz),0],[sin(rotz),cos(rotz),0],[0,0,1]])
    rotz = np.array(rotz)
    
    rot = np.dot(rotx,roty)
    rot = np.dot(rot,rotz);
    
    rot = np.hstack((rot,np.array([[transx],[transy],[transz]])))
    rot = np.vstack((rot,np.array([0,0,0,1])))
    
    return np.dot(rot,x)

def cost2go(ROT):
    
    world_fit = rotate(ROT,sample_c)
    loss = world_fit - sample_r
    loss = np.power(loss,2)
    loss = np.sum(loss)
    
    return loss

def world_pos():
	 # get the rot + trans matrix

	rospy.init_node('camera2world',anonymous = True)

	rospy.Subscriber('camera_view',Vector3,camera_callback )
	rospy.Subscriber('robot_view',Vector3,robot_callback )

	P = np.array([ 1.60593196,  0.10157054, -3.12697484,  0.43266683,  0.51336184,
        0.50048644])#calibrate()
	
	world_pub = rospy.Publisher('world_view', Vector3, queue_size = 10)

	loop_rate = 60
	dt = 1/loop_rate
	rate = rospy.Rate(loop_rate)

	while not rospy.is_shutdown():
		
		xw,yw,zw,ext = rotate(P, np.array([xc,yc,zc,1]))

		world_pub.publish(Vector3(xw,yw,zw))

		rate.sleep()

if __name__ == '__main__':
	settings = termios.tcgetattr(sys.stdin)
	try:
		world_pos()
	except rospy.ROSInterruptException:
		pass