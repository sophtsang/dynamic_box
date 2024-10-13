"""
1) Generate individual 2D boxes for left and right stereo images.
   Use left.png and right.png as test stereo image.
"""

"""
Proposed Regions with YOLOv3: use YOLOv3 to generate 2D bounding boxes 
for all objects in an image.

 1. Input image is divided into NxN grid cells. For each object present on image, one grid cell is responsible for predicting object.

2. Each grid predicts [B] bounding box and [C] class probabilities. And bounding box consist of 5 components (x,y,w,h,confidence)

(x,y) = coordinates representing center of box

(w,h) = width and height of box

Confidence = represents presence/absence of any object
"""


"""
Implement YOLO without the object classification task: any obstacle is an obstacle:
try cv2's selective search: proposes all regions in an image -> associate with obstacle. """

import cv2
import numpy as np

classes = []

def yolo(img):
   # Read weights file (contains pretrained weights which 
   # has been trained on coco dataset) and configuration file 
   # (has YOLOv3 network architecture)
   # [cv2.dnn.readNet] loads the pre-trained YOLO deep learning model.
   net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

   global classes
   # [classes] stores all names of different objects in [coco.names] 
   # that the coco model has been trained to identify 
   with open("coco.names", "r") as f:
      classes = f.read().splitlines()
   
   # Get the name of all layers of the network, then pass to forward pass.
   blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0,0,0), swapRB=True, crop=False)
   net.setInput(blob)

   # Perform forward propogation through OpenCV's DNN: the input image blob is
   # passed through layers of the neural network [networkarch.png] to get output
   # predictions.
   output_layers = net.getUnconnectedOutLayersNames()
   layer_outputs = net.forward(output_layers)
   return layer_outputs

def bounding_box_dim (of, img):
   boxes = []
   confs = []
   class_ids = []

   for output in yolo(img):
      for detect in output:
         scores = detect[5:]
         class_id = np.argmax(scores)
         height, width = img.shape[:2]
         conf = scores[class_id]
         # If image confidence is > 0.3, add bounding box and classification.
         if conf > 0.3:
            center_x = int(detect[0] * width)
            center_y = int(detect[1] * height)
            w = int(detect[2] * width)
            h = int(detect[3] * height)
            # (x,y) are top left coordinates of the bounding box.
            x = int(center_x - w/2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confs.append(float(conf))
            class_ids.append(class_id)

   # Performs non-maximum suppression on duplicate bounding boxes over the same
   # object, keeping only the bounding boxes that have the highest confidence.
   non_dups = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.2)
   colors = np.random.uniform(0, 255, size=(len(boxes), 3))
   for i in non_dups: 
      x, y, w, h = boxes[i]
      label = str(classes[class_ids[i]])
      cv2.rectangle(img, (x,y), (x+w,y+h), colors[i], 2)
      cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[i], 2)
   
   cv2.imshow("box", img)
   cv2.imwrite(of, img)
   cv2.waitKey(0)
   # Return bounding boxes [boxes] to perform ROI align and SSIM with other
   # stereo image: [bounding_box_dim] is performed on both left, right stereo
   # images, and then the resulting bounding boxes of the two are ROI aligned.

bounding_box_dim("boundedleft.png", cv2.imread("left.png"))
bounding_box_dim("boundedright.png", cv2.imread("right.png"))
bounding_box_dim("boundedroad.png", cv2.imread("road.jpg"))
