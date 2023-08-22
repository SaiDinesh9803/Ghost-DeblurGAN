
from predict import Predictor
import cv2
from cv2 import aruco
import numpy as np
import math

predictor = Predictor(weights_path='/home/sai/NN/Ghost-DeblurGAN/trained_weights/fpn_ghostnet_gm_hin.h5')
cap = cv2.VideoCapture(0)
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
detector_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary=dictionary, detectorParams=detector_params)
camera_matrix = np.array([[494.30114815 , 0 , 322.46760411] ,[0 , 493.52032388 , 240.6078525 ] , [0 , 0 , 1]])
camera_distortion = np.array([[ 9.71518504e-03 , 1.17503633e+00 , -1.27224644e-03 , 1.90922858e-03 , -4.38978499e+00]])
marker_length_cm = 2.5  # Length of the marker in centimeters
pixels_per_cm = 1.0 / camera_matrix[0, 0]  # Assuming camera_matrix[0, 0] represents the focal length in pixels


# Calculate the marker size in pixels
marker_size_pixels = marker_length_cm * pixels_per_cm

objPoints = np.array([[-marker_size_pixels / 2, -marker_size_pixels / 2, 0],
                      [-marker_size_pixels / 2, marker_size_pixels / 2, 0],
                      [marker_size_pixels / 2, marker_size_pixels / 2, 0],
                      [marker_size_pixels / 2, -marker_size_pixels / 2, 0]], dtype=np.float32)


while True:
    ret, frame = cap.read()
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    if not ret:
        break
    all_yaw_angles = []
    deblurred_frame = predictor(frame, None)
    corners, ids, rej_corners = detector.detectMarkers(deblurred_frame)
    if ids is not None:
        for i in range(len(ids)):
            marker_corners = corners[i]
            success, r, t= cv2.solvePnP(objectPoints=objPoints, imagePoints=marker_corners, cameraMatrix=camera_matrix, distCoeffs=camera_distortion)
            rot_mat , _ = cv2.Rodrigues(r)
            
            yaw_radians = math.atan2(rot_mat[2][0], rot_mat[0][0])
            yaw_degrees = math.degrees(yaw_radians)
            
            all_yaw_angles.append(yaw_degrees)
            
            
        mean_yaw_angle = sum(all_yaw_angles)/len(all_yaw_angles)
        print(mean_yaw_angle)
    out_image = deblurred_frame.copy()
    out_image = aruco.drawDetectedMarkers(out_image, corners=corners, ids=ids)
    cv2.imshow('output', out_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()