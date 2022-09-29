#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


# python yolo_opencv.py -s true --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
#  python yolo_opencv.py --image sarat.jpeg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt

from cgitb import text
import cv2
import argparse
import numpy as np
from gtts import gTTS
from playsound import playsound





image_filename = None

ap = argparse.ArgumentParser()
ap.add_argument('-s', '--capture', required=True,
                help = 'Yes or No for image capturing')
ap.add_argument('-i', '--image', required=False,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()

if (args.capture == "true"):
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            cv2.imshow("Capturing", frame)

            key = cv2.waitKey(1)
            if key == ord('s'): 
                img_new = cv2.imwrite(filename='saved_img.jpg', img=frame)
                image_filename = 'saved_img.jpg'
                webcam.release()
                # img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                img_new = cv2.imshow("Captured Image", img_new)
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                print("Image saved!")
            
                break
            elif key == ord('q'):
                #Turns off camera
                webcam.release()
                cv2.destroyAllWindows()
                break
            
        except(KeyboardInterrupt):
            #Turns off camera
            webcam.release()
            cv2.destroyAllWindows()
            break

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    for i in net.getUnconnectedOutLayers():
     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if (args.capture == "false"):
    image_filename =  args.image  

image = cv2.imread(image_filename)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4


for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

text1 = "The objects in the image are: "

for i in indices:
    i = i
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    
    fp = open("/home/veeramalli/object-detection-opencv/yolov3.txt")
    for j, line in enumerate(fp):
        if j == class_ids[i]:
            text_1 = line.strip()
            text1 += (text_1+",")
    fp.close()

myobj = gTTS(text=text1, lang='en', slow=False)
  
myobj.save("sarat.mp3")
  


cv2.imshow("object detection", image)
playsound("sarat.mp3")
cv2.waitKey(5000)
    
cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
