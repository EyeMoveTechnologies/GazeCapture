import numpy as np

class Box:
    def __init__(self, topleft, w, h):
        self.topleft = topleft
        self.w = w
        self.h = h
    def corners(self):
        ''' x0,y0, x1,y1'''
        return self.topleft[0], self.topleft[1], self.topleft[0]+self.w, self.topleft[1]+self.h

class Filter:
    def __init__(self, frame_w, frame_h, num_frames=5):

        # Circular queue
        self.num_frames = num_frames

        # Initialize it with a frame the size of half the screen
        self.track_L = np.array([Box((0,0), frame_w/2-50, frame_h)])
        self.track_R = np.array([Box((frame_w/2+50,0), frame_w/2-50, frame_h)])

        self.confidence_thresh = 0.75 # Anything below is discarded
        self.weighing = ['linear', 'exponential_TODO'][0]

        
    
    def update(self, boxes):
        ''' Boxes is array of Box objects'''

        if len(boxes)<2:
            return False, None, None

        box_L, box_R = self.assign_tracks(boxes)

        # Add to existing tracks
        self.track_L = np.append(self.track_L, box_L)
        self.track_R = np.append(self.track_R, box_R)

        return True, box_L, box_R    

    def assign_tracks(self, boxes):
        # TODO: Vectorize all vs. all comparison

        # Keep theh best box for each track
        scores_L = np.zeros((len(boxes)))
        scores_R = np.zeros((len(boxes)))
        for i, box in enumerate(boxes):

            # Score for left track
            score_L = 0
            for box_L in self.track_L:
                score_L += self.IoU(box, box_L)
            score_L /= len(self.track_L) # Normalize
            scores_L[i] = score_L

            # Score for right track
            score_R = 0
            for box_R in self.track_R:
                score_R += self.IoU(box, box_L)
            score_R /= len(self.track_R) # Normalize
            scores_R[i] = score_R

        # Assumption: L and R tracks never overlap. Otherwise, change the following code
        # Keeping it like this for simplicity and speed
        box_L_idx = np.argmax(scores_L)
        box_R_idx = np.argmax(scores_R)

        if box_L_idx == box_R_idx:
            if np.max(scores_L) > np.max(scores_R):
                scores_R[box_R_idx]=-1
                box_R_idx = np.argmax(scores_R)
        
        # Return box_L, box_R
        return boxes[np.argmax(scores_L)], boxes[np.argmax(scores_R)]

        

    def IoU(self, boxA, boxB):
        Acx0, Acy0, Acx1, Acy1 = boxA.corners()
        Bcx0, Bcy0, Bcx1, Bcy1 = boxB.corners()

        xA = max(Acx0, Bcx0)
        yA = max(Acy0, Bcy0)
        xB = min(Acx1, Bcx1)
        yB = min(Acy1, Bcy1)

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (Acx1 - Acx0 + 1) * (Acy1 - Acy0 + 1)
        boxBArea = (Bcx1 - Bcx0 + 1) * (Bcy1 - Bcy0 + 1)
        
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou