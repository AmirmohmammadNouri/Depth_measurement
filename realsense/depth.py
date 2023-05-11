from realsense_camera import *
import cv2 
from mask_rcnn import *

rs = RealsenseCamera()
mrcnn = MaskRCNN()

while True:

    ret , bgr_frame , depth_frame = rs.get_frame_stream()
    # # print(type(depth_frame)) 
    # x_point , y_point = 360,350
    # selected_pixel = depth_frame[x_point,y_point]
    # cv2.circle(bgr_frame,(x_point,y_point),2,(0,0,255),2)
    # cv2.putText(bgr_frame,'distance :{}'.format(selected_pixel),(x_point+10,y_point),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    # print(selected_pixel)
    boxes,classes,contours , centers =  mrcnn.detect_objects_mask(bgr_frame)
    bgr_frame = mrcnn.draw_object_mask(bgr_frame)
    
    mrcnn.draw_object_info(bgr_frame,depth_frame)
    cv2.imshow('bgr frame' , bgr_frame)
    cv2.imshow('depth_frame',depth_frame)
    if cv2.waitKey(1) == ord('q'):
        break
