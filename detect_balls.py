import pickle
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

clf = pickle.load(open("models/ball_clf.pkl", "rb"))

def get_box(img, x, y, w, h):
    '''
    Slices the image around the box coordinates.
    
    Args:
        x: starting x location
        y: starting y location
        w: box width
        h: box height
        
    Returns:
        Image slice containing just the box
    '''
    return(img[y:y+h, x:x+w])

def img_color(ball):
    '''
    Gets the image mean and median colors.
    
    Args:
        ball: masked image of the ball
    
    Returns:
        mean, median
    '''
    colors = []
    for y in range(ball.shape[0]):
        for x in range(ball.shape[1]):
            if ball[y, x, 2] > 1:
                colors.append(ball[y, x, :])
    colors = np.array(colors)
    return np.mean(colors, axis=0), np.median(colors, axis=0)

def circle_mask(ball):
    '''
    Returns masked ball.
    
    Args:
        ball: image of ball
        
    Returns:
        masked image of ball
    '''
    gray = cv2.cvtColor(ball, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=2, minDist=30, param1=50, param2=25, minRadius=7, maxRadius=22)
    if circles is None:
        return ball
    circle = np.uint16(np.around(circles))[0,  :].squeeze()
    
    # If multiple circles detected, take only the first one
    if len(circle) != 3:
        circle = circle[0]
    
    try:
        mask = np.full_like(ball, 255).astype(np.uint8)
        cv2.circle(mask,(circle[0],circle[1]),circle[2],(0,0,0),-1)

        mask = mask[:, :, 0] # Get rid of extra channels
        mask = 255 - mask # flip mask
        return cv2.bitwise_and(ball, ball, mask=mask)
    except:
        return ball

def predict_ball(ball, raw_probs=False):
    '''
    Uses trained RFC to predict ball label.
    
    Args:
        ball: image of ball
        raw_probs: If True, label + probs. for each label returned (defaults to False)
        
    Returns:
        predicted label, and probabilities if raw_probs=True
        
    '''
    ball_mask = circle_mask(ball)
    x = np.append(*img_color(cv2.cvtColor(ball_mask, cv2.COLOR_BGR2HSV)))
    x = x  / np.array((179, 255, 255, 179, 255, 255))
    x = x.reshape(1, -1)
    if raw_probs:
        return clf.predict(x)[0], clf.predict_proba(x)[0]
    return clf.predict(x)[0]


def predict_balls(img, boxes, raw_probs=False):
    '''
    Returns predicted labels for each box in list.
    
    Args:
        img: image containing the balls
        boxes: List-like object containing the boxes around the balls
        raw_probs: If True, label + probs. for each label returned (defaults to False)
        
    Returns:
        list of predicted labels, and probabilities if raw_probs=True
    '''
    balls = [get_box(img, *box) for box in boxes]
    if raw_probs:
        labels=[]
        probs=[]
        for ball in balls:
            label, pr = predict_ball(ball, raw_probs=raw_probs)
            labels.append(label)
            probs.append(pr)
        return labels, probs
    return [predict_ball(ball) for ball in balls]
        

def get_clf():
    '''
    Returns the RFC used
    '''
    return clf