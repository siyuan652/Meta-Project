import mediapipe as mp #import mediapipe
import cv2 # import opencv
import csv
import os
import numpy as np

mp_drawing = mp.solutions.drawing_utils #drawing the helpers
mp_holistic = mp.solutions.holistic #mediapipe solutions

cap = cv2.VideoCapture(0)
#INITIATE HOLISTIC MODEL

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        print (frame)
        
        #RECOLOR FEED
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        image = cv2.flip(image,1)
        
        #MAKE DETECTIONS
        results = holistic.process(image)
        print(results.face_landmarks)

        #FACE_LANDMARKS, POSE_LANDMARKS, LEFT_HAND_LANDMARKS, RIGHT_HAND_LANDMARKS

        #RECOLOR IMAGE BACK TO BGR FOR RENDERING
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #DRAW FACE LANDMARKS
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
                                 )

        #DRAW RIGHT HAND LANDMARKS
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                                 )

        #DRAW LEFT HAND LANDMARKS
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                                 )

        #Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark)

        #DEFINE COORDINATES AND CSV FILE
        landmarks = ['class']
        for val in range(1, num_coords+1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

        class_name = "happy"
        #class_name = "neutral"
        #class_name = "sad"
        #class_name = "angry"
        #class_name = "depressed"

        #EXPORT COORDINATES
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            
            # Concate rows
            row = pose_row+face_row
            
            # Append class name 
            row.insert(0, class_name)
            
            #Add class and joint label to CSV (Run this first)
            #with open('coords.csv', mode='w', newline='') as f:
                #csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                #csv_writer.writerow(landmarks) 

            #EXPORT DATA to CSV (Then run this first)
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row) 

        except:
           pass

        cv2.imshow('Holistic Model Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()