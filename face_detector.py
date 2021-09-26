#import the required library
import cv2
import mediapipe as mp
import time

# we are going to use classes in this code

# Initialize the module to begin with face detection
class FaceDetector:
    def __init__(self, minDetectionConfidence=0.60):
        self.minDetectionConfidence = minDetectionConfidence #to detect face with "60%" accuracy
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.facedetection = self.mpFaceDetection.FaceDetection(self.minDetectionConfidence)


    # apply the module
    def findfaces(self, img):
        #convert the BGR image to RGB for face detection
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.facedetection.process(imgRGB)

        # ---------------------------------------------------------------#
        bbox = [] #stores the co-ordinate and detection score of the face
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                #self.mpDraw.draw_detection(img, detection)  # uncomment to detect eyes nose ears mouth points
                bounding_box_points = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                #bounding box co-ordinates
                bounding_box = int(bounding_box_points.xmin * iw), int(bounding_box_points.ymin * ih), \
                               int(bounding_box_points.width * iw), int(bounding_box_points.height * ih)
                # append the bounding_box and detection
                bbox.append([id, bounding_box, detection.score])
                # draws bounding box around the face
                cv2.rectangle(img, bounding_box, (255, 0, 255), 2)
                #puts the accuracy of face at the top of the bounding box you can also change the font style,size and colour
                cv2.putText(img, f'{int(detection.score[0] * 100)}%',(bounding_box[0], bounding_box[1] - 20),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        #return img and bbox to use it in Main
        return img, bbox


# opens the camera and starts processing the the input images or video to detect faces
def Main():
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    # start time to to calculate fps
    pTime = 0
    # call the FaceDetector function and save in detector variable
    detector = FaceDetector()
    while True:
        success, video = cap.read()
        video, bbox = detector.findfaces(video)
        # FPS calculating
        cTime = time.time()
        FPS = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(video, f'FPS:{int(FPS)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("face_detector", video)
        cv2.waitKey(1)


#Calls the funtion
Main()