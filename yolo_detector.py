import os
import cv2
import numpy as np
import pandas as pd

class YoloDetector:
    def __init__(self, cfg_file, weights_file, classes, conf_level=.5, threshold=.3):
        self.conf_level = conf_level
        self.threshold = threshold
        self.classes = classes
        
        # Initialize colors for each class
        np.random.seed(42)
        
        # Load neurel net weights
        self.net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)        
        layers = self.net.getLayerNames()
        self.layers = [layers[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
    def detect_image(self, img, useNMS=True):
        '''
        Runs detector on image
        
        Args:
            img: image containing objects
            useNMS: If True, use non-maxima suppresion on overlapping bounding boxes
            
        Returns:
            Boxes, confidence levels, classesIDs
            
        '''
        self.h_, self.w_ = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.layers)
        
        boxes = []
        confidences = []
        classIDs = []
        
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]  # Extract confidence scores
                classID = np.argmax(scores)
                conf = scores[classID]
                
                if conf > self.conf_level:
                    box = detection[:4] * np.array([self.w_, self.h_, self.w_, self.h_])  # Rescale yolo img percent to pixels
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(conf))
                    classIDs.append(classID)
         
        if useNMS:
            # Perform Non-maxima suppression to remove overlappying bounding boxes and keep the most confident boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_level, self.threshold)
            new_boxes = []
            new_confidences = []
            new_classIDs = []
            if len(idxs) > 0:
                for i in idxs.flatten():
                    new_boxes.append(boxes[i])
                    new_confidences.append(confidences[i])
                    new_classIDs.append(classIDs[i])
                
            return new_boxes,new_confidences, new_classIDs
        
        return boxes, confidences, classIDs  
    
    def draw_boxes(self, img, boxes, conf, labels):
        '''
        Draws bounding boxes on image.
        
        Args:
            img: image with objects
            boxes: location of objects
            conf: confidence level of objects
            labels: object labels
            
        Returns:
            New image with bounding boxes drawn
        '''

        img_cp = img.copy()

        for i in range(len(boxes)):
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            cv2.rectangle(img_cp, (x,y), (x+w, y+h), (0,255,0))
            text = f"Ball {labels[i]}: {round(conf[i], 3)}"
            cv2.putText(img_cp, text, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        return img_cp

