import cv2
import numpy as np
import os
import time

cap = cv2.VideoCapture('2.mp4')

ret, frame = cap.read()
prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame)
hsv[...,1] = 255

i = 0
start_time = time.time()
instance = time.time()
while(cap.isOpened()):
    ret, frame = cap.read()

    next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = np.zeros_like(prvs)
    flow = cv2.createOptFlow_DualTVL1().calc(prvs,next,flow)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imwrite('/home/edward/MAIN DATA/Output/RGB/opticalrgb'+str(i)+'.png',rgb)
    
    iter_time = time.time() - instance
    instance = time.time() 
    print("Number of frames generated: ",i+1,"    Time taken: ",iter_time)

    i = i+1
    prvs = next

print("Total Time: ",time.time()-start_time)
print("Average Time per frame: ",(time.time()-start_time)/i)
cv2.destroyAllWindows()