import cv2
import sys
import numpy as np
from eye_tracking_filter import Filter, Box

# if len(sys.argv) < 2:
#     cascPath = '/home/eyemove/fydp/GazeCapture/pytorch/haar.xml'
#     eyeCascPath = '/home/eyemove/fydp/GazeCapture/pytorch/haar_eye.xml'
# else:
cascPath = sys.argv[1]
eyeCascPath = sys.argv[2]

print('Face cascade: ', cascPath)
print('eye cascade: ', eyeCascPath)
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyeCascPath)

cam = cv2.VideoCapture("v4l2src device=/dev/video0 ! videorate ! video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 ! nvvidconv ! video/x-raw(memory:NVMM) ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=2 drop=true", cv2.CAP_GSTREAMER)

DETECT_FACES = False


w, h = 224,224
fil = Filter(w, h)

while True:

    ret, frame = cam.read()

    if ret:

        # Rescale
        center = frame.shape[0]/2, frame.shape[1]/2
        x = center[1] - w/2
        y = center[0] - h/2
        # print(type(frame))
        frame = frame[int(y):int(y+h), int(x):int(x+w)]

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        if DETECT_FACES:
            faces, neighs, weights = faceCascade.detectMultiScale3(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE,
                outputRejectLevels = True
            )

            print ("Found {} faces!".format(len(faces)))

            # Draw a rectangle around the faces
            for (fx, fy, fw, fh) in faces:
                box = Box((fx, fy), fw, fh)
                x0,y0,x1,y1 = box.corners()
                cv2.rectangle(frame,(x0,y0),(x1,y1),(0,0,255),2)
                print("Added face rectangle")
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # # eyes, neighs, weights = eyeCascade.detectMultiScale3(gray[y:y+h, x:x+w])
                # eyes, neighs, weights = eyeCascade.detectMultiScale3(
                #     gray,
                #     scaleFactor=1.1,
                #     minNeighbors=5, 
                #     minSize=(30, 30),
                #     flags = cv2.CASCADE_SCALE_IMAGE,
                #     outputRejectLevels = True
                # )

                # for (ex,ey,ew,eh) in eyes:
                #     cv2.rectangle(frame,(x0,y0),(x1,y1),(0,255,0),2)
                #     cv2.rectangle(frame[y:y+h, x:x+w],(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
                #     cv2.putText(frame[y:y+h, x:x+w], "{}".format(), )
        if True:
            eyes, neighs, weights = eyeCascade.detectMultiScale3(
                gray,
                scaleFactor=1.1,
                minNeighbors=5, 
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE,
                outputRejectLevels = True
            )

            boxes = []
            for i, (ex,ey,ew,eh) in enumerate(eyes):
                box = Box((ex, ey), ew, eh)
                boxes += [box]

            # Apply temporal filter to predictions
            r, boxL, boxR = fil.update(boxes)
            if not r:
                continue

            x0,y0,x1,y1 = boxL.corners()
            cv2.rectangle(frame,(x0,y0),(x1,y1),(255,0,0),2)

            x0,y0,x1,y1 = boxR.corners()
            cv2.rectangle(frame,(x0,y0),(x1,y1),(0,255,0),2)

            # print(weights[i])
            # cv2.putText(frame, "{:.1f}".format(weights[i][0]),(ex, ey), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255) )
            
            
        cv2.imshow("Faces found", frame[:,::-1])
        cv2.waitKey(1)