import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh()

while True:
    _, frame = cam.read()
    rgb_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmarks_points = output.multi_face_landmarks
    frame_h , frame_w , _= frame.shape
    if(landmarks_points):
        landmarks = landmarks_points[0].landmark
        for landmark in landmarks:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame , (x,y), 1 , (0,255,0))
    cv2.imshow("Eye Controlled Mouse" , frame)
    cv2.waitKey(1)
