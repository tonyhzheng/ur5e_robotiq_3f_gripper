#!/usr/bin/env python
# simulate the tracker without camera involved 
import numpy as np
from numpy import sin, cos 
import cv2
import time
import numpy as np
from math import pi
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32
import rospy
import pyrealsense2 as rs

max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
low_H = 68
low_S = 0
low_V = 0
high_H = 255
high_S = 40
high_V = 40
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low R'
low_S_name = 'Low G'
low_V_name = 'Low B'
high_H_name = 'High R'
high_S_name = 'High G'
high_V_name = 'High B'

Width = 640
Height = 480

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

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)

# cv2.namedWindow(window_capture_name)
cv2.namedWindow(window_detection_name)
cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)

class Tracker:
    """
    A basic color tracker, it will look for colors in a range and
    create an x and y offset valuefrom the midpoint
    """

    def __init__(self, height, width, color_lower, color_upper):
        self.color_lower = color_lower
        self.color_upper = color_upper
        self.midx = int(width / 2)
        self.midy = int(height / 2)
        self.xoffset = 0
        self.yoffset = 0
        # self.depth = 0

    def draw_arrows(self, frame):
        """Show the direction vector output in the cv2 window"""
        return frame

    def track(self, frame):
        """Simple HSV color space tracking"""
        # resize the frame, blur it, and convert it to the HSV
        # color space
        hsv = cv2.GaussianBlur(frame, (11, 11), 0)
        # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        # mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
        # (low_H, low_S, low_V), (high_H, high_S, high_V) (0, 0, 0), (7, 255, 255)
        mask = cv2.inRange(hsv, (low_V, low_S, low_H), (high_V, high_S, high_H))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow('mask', mask)
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        img, cnts, trash= cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        # print(cnts)
        # cnts = cnts[0] # this is a set of points
        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid

            c = max(cnts, key=cv2.contourArea)
            # c = max(cnts, cv2.contourArea(cnts))
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            # print( x, y, radius) # "x = %d, y = %d, r = %d" % 
            M = cv2.moments(c) 
            # print("XXXXXX", c)

            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            # if not (M["m00"]==0):
            #    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # else:
            #    center = (0,0)
        
            # only proceed if the radius meets a minimum size
            if radius > 1:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                self.xoffset = int(center[0])

                self.yoffset = int(center[1])
                
            else:
                self.xoffset = 0
                self.yoffset = 0
                # self.depth = 0
        else:
            self.xoffset = 0
            self.yoffset = 0
            # self.depth = 0
        return self.xoffset, self.yoffset # , self.depth    


def main():
    
    red_lower = (0, 104, 118)
    red_upper = (23, 255, 255)

    x_off = [0]
    y_off = [0]

    level = 10
    search_num = level**2

    for i in range(-10,11):
        for j in range(-10,11):
            x_off.append(i)
            y_off.append(j)

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, Width, Height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, Width, Height, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Intrinsic Matrix
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # print("Depth Scale is: " , depth_scale)

    # align depth stream to GRB stream
    align_to = rs.stream.color
    align = rs.align(align_to)

    # create ball tracker for dilter the color
    redtracker = Tracker(Width, Height, red_lower, red_upper)

    # create camera node
    rospy.init_node('position_pub',anonymous = True)

    depth_pub = rospy.Publisher("depth",Float32, queue_size = 1)
    depth_state_pub = rospy.Publisher("depth_error",Float32, queue_size = 1)
    rate_pub = rospy.Publisher("loop_rate",Float32, queue_size = 1)

    camera_pub = rospy.Publisher('camera_view', Vector3, queue_size = 10)
##### robot_pub = rospy.Publisher('robot_view',Vector3,queue_size = 10)

    # loop rate
    loop_rate = 30
    dt = 1/loop_rate
    rate = rospy.Rate(loop_rate)

    while not rospy.is_shutdown():

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        # depth_frame_ = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # if not depth_frame_ or not color_frame:
        #    continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_intrinsics = rs.video_stream_profile(
        aligned_frames.profile).get_intrinsics()

        # X-480 Y-640
        X, Y= redtracker.track(color_image)
        color_image = redtracker.draw_arrows(color_image)

        #X = 720
        #Y = 550
       

        Depth = rs.depth_frame.get_distance(aligned_depth_frame, X, Y)

        X_ = X
        Y_ = Y

        TT = 0
        jj = 0
        while (Depth==0):

            X = X_ + x_off[jj]
            Y = Y_ + y_off[jj]

            Depth = rs.depth_frame.get_distance(aligned_depth_frame, X, Y)

            if (Depth!=0):
                    depth_pub.publish(Depth)

            jj = jj+1

            if (Depth!=0) or (jj==search_num):
                if (jj==search_num) and (Depth==0):
                    depth_state_pub.publish(0)
                break
                #
        # Depth = depth_image[X,Y]
        # print('X =  %.10f, Y = %.10f, Z = %.10f' % (X, Y, Depth))

        # print(Depth)
        # Y = 240
        # X = 320
        X, Y, Z= rs.rs2_deproject_pixel_to_point(depth_intrinsics, [X, Y], Depth)
        if Depth!=0:
            camera_pub.publish(Vector3(X,Y,Z))
        rate_pub.publish(0)
## X, Y, Z, trash = rotate(np.array([pi/2, 0,0,1,2,3]),np.array([X,Y,Z,1]))

## robot_pub.publish(Vector3(X,Y,Z))
        # print('real_depth: ',(Y, X, Depth))
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Depth = depth_image[X, Y]

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # images = color_image
        # Show images
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(images,'Y: %.4f, X: %.4f Depth: %.4f' % (Y, X, Z),(10,450), font, 0.8,(0,255,0),2,cv2.LINE_AA)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        key = cv2.waitKey(30)
        if key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        rate.sleep()
        # cv2.waitKey(1)

        # Stop streaming
    pipeline.stop()
        

if __name__ == '__main__':
    main()
