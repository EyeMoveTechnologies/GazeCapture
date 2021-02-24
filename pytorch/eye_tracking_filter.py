import numpy as np

class Box:
    def __init__(self, center):
        self.center = center
        self.w = w
        self.h = h

class Filter:
    def __init__(self, num_frames=5):

        # Circular queue
        self.num_frames = num_frames
        self.i = 0

        self.track_L = np.zeros((self.num_frames,))
        self.track_R = np.zeros((self.num_frames,))

        self.confidence_thresh = 0.75 # Anything below is discarded
        self.weighing = ['linear', 'exponential_TODO'][0]

        
    
    def update(self, boxes):

        self.assign_track(boxes)

        self.i += 1
        self.i = i%self.num_frames

        filtered_box = self.filter()
        return filtered_box


    def assign_track(self, boxes):
        # TODO: Vectorize all vs. all comparison

        # Keep theh best box for each track
        scores_L = np.zeros(len(boxes))
        scores_R = np.zeros(len(boxes))
        for i, box in enumerate(boxes):

            # Score for left track
            score_L = 0
            for box_L in self.track_L:
                score_L += self.IoU(box, self.track_L)
            score_L /= len(self.track_L) # Normalize
            scores_L[i] = score_L

            # Score for right track
            score_R = 0
            for box_R in self.track_R:
                score_R += self.IoU(box, self.track_R)
            score_R /= len(self.track_R) # Normalize
            scores_R[i] = score_R

        s = np.vstack((scores_L, scores_R))
        L_R = np.argmax(s, axis=0)

        
        for j in L_R:
            if j == 0:
                candidate_Ls.append()

        

    def IoU(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou