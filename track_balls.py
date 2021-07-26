import cv2
import numpy as np
import pandas as pd
from yolo_detector import YoloDetector
from detect_balls import *
from table import Table

class BallTracker():
    
    def __init__(self, yolo, start_frame=0, num_balls=10):
        '''
        Instantiates ball tracker.
        
        Args:
            yolo: YoloDetector object
            start_frame: frame to start indexing
            num_balls: number of balls to look for
        '''
        self.prev_balls = None
        self.cur_balls = [None for _ in range(num_balls)]
        self.hist = {i:[] for i in range(num_balls)}
        self.hist['orientation'] = []
        self.times = []
        self.yolo = yolo
        self.current = start_frame - 1
        self.table = Table()
        
    def update(self, img, draw_balls=True):
        '''
        Runs detection and tracking. Records if balls have moved.
        
        Args:
            img: image to run detection on
            draw_balls: If True, return image with locations marked
            
        Returns:
            Copy of image
        '''
        self.current += 1
        
        orientation, table = self.table.find_table(img)

        if table is None:
            return img
        
        boxes, confidences, _ = self.yolo.detect_image(img)
        labels, probs = predict_balls(img, boxes, raw_probs=True)
        
        self.prev_balls = self.cur_balls
        balls = [None for _ in self.cur_balls]
        
        cur_prob = [0 for _ in self.cur_balls]
        
        for label, box, prob in zip(labels, boxes, probs):
            if prob[label] > cur_prob[label]:
                balls[label] = self.ball_center(box)
                cur_prob[label] = prob[label]
        

        self.cur_balls = balls
        for i, ball in enumerate(self.cur_balls):
            transformed_ball = self.table.transform_point(ball, orientation)
            self.hist[i].append(transformed_ball)
        self.times.append(self.current)
        self.hist['orientation'].append(orientation)
        
        if draw_balls:
            img_cp = img.copy()
            for ball, prob in zip(self.cur_balls, cur_prob):
                if ball is not None:
                    cv2.circle(img_cp, tuple(ball), 4, (0,255,0), -1)
                    
            return img_cp
            
        return img
        
    def ball_moved(self, new_balls, min_dist = 5):
        '''
        Returns true if a ball moved more than 5 pixels in either x or y direction
        '''
        if self.prev_balls is None:
            return True
        
        for prev, cur in zip(self.prev_balls, new_balls):
            if (prev is None) and (cur is not None):
                return True
            if (prev is None) or (cur is None):
                continue               
            diff = np.absolute(cur - prev)
            if np.any(diff > min_dist):
                return True
            
        return False
        
        
    def ball_center(self, box):
        '''
        Returns the point at the center of the box as numpy array
        '''
        x, y, h, w = box
        return np.array([x + w//2, y + h//2])
    

    
    def save(self, out_file):
        '''
        Saves the ball movement to csv
        '''
        df = pd.DataFrame(self.hist, index=self.times)
        df.to_csv(out_file)