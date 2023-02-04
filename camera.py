import cv2 as cv
import numpy as np

class Camera:

    def __init__(self, window):
        """
        :rtype: object
        """
        self.window = window
        self.camera = cv.VideoCapture(self.window)
        if not self.camera.isOpened():
            raise ValueError("Unable to open camera!")

        self.width = self.camera.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.camera.get(cv.CAP_PROP_FRAME_HEIGHT)
        
        # chessboard size
        self.chessboard_size = (9, 6)

        # termination criteria for the cornerSubPix() function
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points
        self.objp = np.zeros((self.chessboard_size[0]*self.chessboard_size[1],3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1,2)

        # arrays to store object points and image points
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.

    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()

    def get_frame(self):
        if self.camera.isOpened():
            
            ret, frame = self.camera.read()
            if ret == True:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                ret, corners = cv.findChessboardCorners(gray, self.chessboard_size, None)
                
                if ret == True:
                    self.objpoints.append(self.objp)
                    corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
                    self.imgpoints.append(corners2)
                    frame = cv.drawChessboardCorners(frame, self.chessboard_size, corners2, ret)
                    
                return (ret, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            else:
                return (ret, None)

cv.destroyAllWindows()
