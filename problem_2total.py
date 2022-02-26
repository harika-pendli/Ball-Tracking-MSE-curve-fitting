# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 23:54:13 2022

@author: pendl
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# =============================================================================
# Base_path_in="C:\\Users\\pendl\\Desktop"
# Base_path_out="C:/Users/pendl/Desktop"
# #video_number=input("Please enter 1 for video1 or 2 for video2:")
# 
# path_video1=Base_path_in + "\\673_hw1\\ball_video1.mp4"
# path_out1=Base_path_out+"/673_hw1/ball_video1_frames"
# base_path_1=Base_path_out+"/673_hw1/ball_video1_frames"
#     
# path_video2=Base_path_in + "\\673_hw1\\ball_video2.mp4"
# path_out2=Base_path_out+"/673_hw1/ball_video2_frames"
# base_path_2=Base_path_out+"/673_hw1/ball_video2_frames"
# ==========================================================================
Base_path_in="C:\\Users\\pendl\\Desktop"
Base_path_out="C:/Users/pendl/Desktop"
bo=False

while (bo== False):
    video_number =int(input("Enter video 1 or video 2. Type 1 or 2: "))
    if (video_number ==1):
        bo=True
        path_in=Base_path_in + "\\673_hw1\\ball_video1.mp4"
        path_out=Base_path_out+"/673_hw1/"
    elif(video_number==2):
        bo=True
        path_in= Base_path_in +"\\673_hw1\\ball_video2.mp4"
        path_out=Base_path_out + "/673_hw1/"
    else:
        print("Type 1 or 2!")
        

    
    

    



path_in="C:\\Users\\pendl\\Desktop\\673_hw1\\ball_video1.mp4"
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(path_in)

# Check if camera opened successfully

if (cap.isOpened()== False): 
  print("Error opening video stream or file")


# Read until video is completed
datapoints=[]
count=0
while(cap.isOpened()):
  # Capture frame-by-frame
  
  ret, image = cap.read()
  
  if ret == True:

    # Display the resulting frame
    cv2.imshow('Frame',image)
    path_out="C:/Users/pendl/Desktop/673_hw1/"
    cv2.imwrite(os.path.join(path_out ,"frame%d.jpg" % count ), image)
    count=count+1
    print(np.shape(image))
    # Press Q on keyboard to  exit
    
    #cv2.imshow("img",image)
    result = image.copy()
       
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
         
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
         
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])
        
    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)
         
    full_mask = lower_mask + upper_mask;
         
    result = cv2.bitwise_and(result, result, mask=full_mask)
         
    coordinates=[]
    for i in range(0,np.shape(full_mask)[0]):
        for j in range(0,np.shape(full_mask)[1]):
            if(full_mask[i][j] != 0):
                coordinates.append((i,j))
                    
                        
                
                    
    x= (coordinates[0][0] + coordinates[-1][0]) /2
    y= (coordinates[0][1] + coordinates[-1][1]) /2
         
                    
    datapoints.append((x,y))
        
        
                    
    dp=np.array(datapoints)
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break


# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
#print (dp)

#transform image coordinates to cartesian coordinates:
x_i = dp[:, 0]
y_i = dp[:, 1]

x_c = y_i
y_c = np.shape(full_mask[0]) - x_i

o = np.ones(x_c.shape)

z = np.vstack((np.square(x_c), x_c, o)).T
t1 = np.dot(z.transpose() , z)
t2 = np.dot(np.linalg.inv(t1), z.transpose())

A = np.dot(t2, y_c.reshape(-1, 1))

x_min = np.min(x_c)
x_max = np.max(x_c)

x_curve = np.linspace(x_min-10, x_max+10, 300) 
o_curve = np.ones(x_curve.shape)
z_curve = np.vstack((np.square(x_curve), x_curve, o_curve)).T
#print("z_curve shape = ", z_curve.shape)
#print("coef shape = ", coef.shape)
#print("x_curve: ",x_curve)
#print("z_curve: ",z_curve)

y_curve = np.dot(z_curve, A)
#print("y_curve shape = ", y_curve.shape)
#print("y_curve :",y_curve)

plt.figure()
plt.plot(x_c, y_c, 'ro', x_curve, y_curve, '-b')
plt.show()
    
plt.scatter(x_c,y_c)
plt.show()
    

