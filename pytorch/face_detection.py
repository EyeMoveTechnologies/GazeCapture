import cv2
import sys
import numpy as np
# from eye_tracking_filter import Filter

cascPath = sys.argv[1]
eyeCascPath = sys.argv[2]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyeCascPath)

cam = cv2.VideoCapture("v4l2src device=/dev/video0 ! videorate ! video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 ! nvvidconv ! video/x-raw(memory:NVMM) ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true", cv2.CAP_GSTREAMER)

DETECT_FACES = False

while True:

    ret, frame = cam.read()

    if ret:

        # Rescale
        w, h = 224,224
        center = frame.shape[0]/2, frame.shape[1]/2
        x = center[1] - w/2
        y = center[0] - h/2
        print(type(frame))
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
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                eyes, neighs, weights = eyeCascade.detectMultiScale3(gray[y:y+h, x:x+w])

                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(frame[y:y+h, x:x+w],(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
                    cv2.putText(frame[y:y+h, x:x+w], "{}".format(), )
        else:
            eyes, neighs, weights = eyeCascade.detectMultiScale3(
                gray,
                scaleFactor=1.1,
                minNeighbors=5, 
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE,
                outputRejectLevels = True
            )

            for i, (ex,ey,ew,eh) in enumerate(eyes):
                cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(np.random.randint(0, 255),np.random.randint(0, 255),np.random.randint(0, 255)),2)
                print(weights[i])
                cv2.putText(frame, "{:.1f}".format(weights[i][0]),(ex, ey), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255) )

            
        cv2.imshow("Faces found", frame[:,::-1])
        cv2.waitKey(1)