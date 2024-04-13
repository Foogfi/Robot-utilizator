from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from serial.tools import list_ports
import time
import pydobot

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(1)
flag = True
while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    if class_name[2:] == "Class 1" and np.round(confidence_score * 100) >= 99:
        flag = False

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)
    break
    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break






if flag:
    available_ports = list_ports.comports()
    print(f'available ports: {[x.device for x in available_ports]}')
    port = available_ports[0].device
    port1 = available_ports[1].device
    device1 = pydobot.Dobot(port=port1, verbose=False)

    device = pydobot.Dobot(port=port, verbose=False)

    (x, y, z, r, j1, j2, j3, j4) = device.pose()
    print(f'x:{x} y:{y} z:{z} j1:{j1} j2:{j2} j3:{j3} j4:{j4}')

    device.speed(10000, 10000)

    device1.speed(10000, 10000)

    device1.move_to(217, -3, 137, -1, wait=False)

    device.move_to(227, 0, 119, -3, wait=False)

    device.move_to(212, -2, 22, -4, wait=False)

    device.move_to(285, -17, -3, -6, wait=False)

    device.move_to(216, 8, 21, -1, wait=False)

    device.move_to(269, 14, -22, 0, wait=False)

    device1.move_to(203, -5, 71, -1.4, wait=False)
    time.sleep(2.4)
    device1.move_to(262, -3, 53, -1, wait=False)

    device.close()