'''
Camera Classifier v0.1 Alpha
Copyright (c) NeuralNine
Instagram: @neuralnine
YouTube: NeuralNine
Website: www.neuralnine.com
'''

import cv2 as cv


class Camera:

    def __init__(self):
        """

        :rtype: object
        """
        self.camera = cv.VideoCapture(self.window)
        if not self.camera.isOpened():
            raise ValueError("Unable to open camera!")

        self.width = self.camera.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.camera.get(cv.CAP_PROP_FRAME_HEIGHT)

    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()

    def get_frame(self):
        if self.camera.isOpened():
            
            ret, frame = self.camera.read()
# If found, add object points, image points (after refining them)
    if ret == True:

        return (ret, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        # Draw and display the corners
       
        cv.imshow('self', self)
        cv.waitKey(1000)
else:
    return (ret, None)

cv.destroyAllWindows()

            
             
