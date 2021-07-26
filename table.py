import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model

class Table:
    
    # Labels for the keras model
    labels = {
        0:'End1',
        1:'End2',
        2:'Horizontal',
        3:'Logo',
        4:'Rules',
        5:'Vertical'
    }
    
    # Shape of images model was trained on
    input_shape = (854, 480, 3)
    
    # Corners for perspective transform
    h_corners_new = [(400,0),(400,800),(0,800),(0,0)]
    v_corners_new = [(0,0),(400,0),(400,800),(0,800)]
    
    # Pre-computed old corners for perspective transform
    h_corners = [(420, 385), (1525, 385), (1730, 830), (215, 830)]
    v_corners = [(760, 370), (1190, 370), (1415, 885), (500, 885)]
    
    def __init__(self):
        self.v_dim = None
        self.h_dim = None
        self.v_corners = None
        self.h_corners = None
        self.h_m = None
        self.v_m = None
        self.clf = load_model('models/table_model_extended.h5')

    def get_table_dim(self, img, orientation):
        '''
        Gets the table dimensions. Looks for cached results first.
        
        Also, sets the corner attributes and transform matrix attributes for each
        table orientation during calls to this method.
        
        Args:
            img: image containing table
            orientation: the table orientation, 'Horizontal' or 'Vertical'
            
        Returns:
            Bounding rectangle around the table
        '''
        if orientation not in ['Horizontal', 'Vertical']:
            return None

        if (orientation == 'Horizontal') and self.h_dim:
            return self.h_dim

        if (orientation == 'Vertical') and self.v_dim:
            return self.v_dim

        img_blur = cv2.medianBlur(img, 17)
        gray_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray_blur, 110, 255, cv2.THRESH_BINARY)
        thresh = np.uint8(thresh)

        kernel = np.ones((7,7), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=6)

        sure_bg = cv2.dilate(morph, kernel, iterations=3)

        dist_transform = cv2.distanceTransform(morph, cv2.DIST_L2, 5)

        ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

        sure_fg = np.uint8(sure_fg)

        unknown = cv2.subtract(sure_bg, sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown==255] = 0

        markers = cv2.watershed(img, markers)

        contours, hierarchy = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        table = cv2.boundingRect(contours[0])
        corners = self._find_corners(contours[0])

        if orientation == 'Horizontal':
            self.h_dim = table
            self.h_corners = corners
            self.h_m = cv2.getPerspectiveTransform(np.array(self.h_corners, dtype=np.float32), np.array(Table.h_corners_new, dtype=np.float32))

        elif orientation == 'Vertical':
            self.v_dim = table
            self.v_corners = corners
            self.v_m = cv2.getPerspectiveTransform(np.array(self.v_corners, dtype=np.float32), np.array(Table.v_corners_new, dtype=np.float32))

        return table

    def find_table(self, img, pre_compute=True):
        '''
        Returns a bounding box around the table or None if no table found.
        
        Args:
            img: Image
            pre_compute: If True, use pre-computed values for corners and transofrmation matrices
            
        Returns:
            the label for the table, and the dimensions of the table
        '''
        img_cp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cp = cv2.resize(img_cp, self.input_shape[:2])
        img_cp = np.expand_dims(img_cp, axis=0)
        img_cp = img_cp / 255

        pred = np.argmax(self.clf.predict(img_cp), axis=1)[0]

        if (pred != 2) and (pred != 5):
            return None, None
        
        if pre_compute:
            self.h_dim = (0,0,0,0)
            self.h_m = cv2.getPerspectiveTransform(np.array(Table.h_corners, dtype=np.float32), np.array(Table.h_corners_new, dtype=np.float32))
            
            self.v_dim = (0,0,0,0)
            self.v_m = cv2.getPerspectiveTransform(np.array(Table.v_corners, dtype=np.float32), np.array(Table.v_corners_new, dtype=np.float32))
            
            return Table.labels[pred], (0,0,0,0)
        
        dim = self.get_table_dim(img, Table.labels[pred])

        return Table.labels[pred], dim
    
    def _find_corners(self, contour):
        '''
        Given a contour of the table, finds the corners of the table.
        '''
        box = contour.squeeze()

        b_left = box[:, 0].min(), box[:, 1].max()
        b_right = box[:, 0].max(), box[:, 1].max()    

        top_edge_y = box[:, 1].min()
        top_edge_x = [x for (x,y) in box if y < top_edge_y + 10]

        t_left = min(top_edge_x), top_edge_y
        t_right = max(top_edge_x), top_edge_y

        return [t_left, t_right, b_right, b_left]           
    
    def transform_point(self, pt, orientation):
        '''
        Returns transformed point, based on table orientation.        
        find_table must be called first.
        
        Args:
            pt: point to transform
            orientation: 'horizontal' or 'vertical'
            
        Returns:
            transformed point
        '''
        if pt is None:
            return None
        
        m = None
        if orientation == 'Horizontal':
            m = self.h_m
        elif orientation == 'Vertical':
            m = self.v_m
        else:
            return None
        
        x, y = pt

        new_x = (m[0,0]*x + m[0,1]*y + m[0,2]) / (m[2,0]*x + m[2,1]*y + m[2,2])
        new_y = (m[1,0]*x + m[1,1]*y + m[1,2]) / (m[2,0]*x + m[2,1]*y + m[2,2])

        return int(new_x), int(new_y)
    
    def transform_table(self, img, orientation):
        '''
        Returns transformed table img, based on table orientation.        
        find_table must be called first.
        
        Args:
            img: image containing table
            orientation: 'horizontal' or 'vertical'
            
        Returns:
            transformed table image
        '''
        m = None
        if orientation == 'Horizontal':
            m = self.h_m
        elif orientation == 'Vertical':
            m = self.v_m
        else:
            return None
        
        return cv2.warpPerspective(img, m, (400,800))
