from flask import Flask
from flask import render_template
from flask import Response
import cv2
import cv2
import mediapipe as mp
import random
import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from keras.preprocessing.image import load_img, img_to_array



app = Flask(__name__)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
#                                       "haarcascade_frontalface_default.xml")
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def generate():
    while True:
        ret, frame = cap.read()
        with mp_hands.Hands(
                static_image_mode=False,  # Video
                max_num_hands=1,
                min_detection_confidence=0.5) as hands:

            height, width, _ = frame.shape
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if ret:
                if results.multi_hand_landmarks is not None:  # para puntos de referencia de la mano
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(
                                color=(0, 255, 255), thickness=3, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=1))
                else:
                    font = cv2.FONT_HERSHEY_COMPLEX  # FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        frame, 'NO SE DETECTA LA MANO', (50, 50), font, 1.3, (255, 0, 204), 2, cv2.LINE_8)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag:
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                  bytearray(encodedImage) + b'\r\n')


@app.route("/",methods=['GET'])
def index():
    return render_template("indexOpe.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=False)

# cap.release()

@app.route("/operaciones")
def capturar(): 
    salida = cv2.VideoWriter('webCam.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (640,480)) 
    while (cap.isOpened()):
        ret, frame = cap.read() #captura de fotogramas
        cv2.imshow('webCam',frame)
        salida.write(frame)
        if (cv2.waitKey(1) == ord('s')): #s salir del blucle
            break
        salida.release()
    return render_template("operaciones.html")

def realizarOperacion():
    # opcion= 1
    numero1= 1
    numero2= 1
    while(id<5):
        if(id=="sumar"):
            resultado=numero1 + numero2
            break
        if(id=="restar"):
            resultado=numero1 - numero2
            break
        if(id==3):
            resultado=numero1 * numero2
            break
        if(id==1):
            resultado=numero1 / numero2
            break
    if(id==5):
        return Response(index())
    return resultado


# def obtenerNumero():
#     numero= random()
#     # return numero

# @app.route("/numero")
# def numeroAleatorio():
#     return Response(obtenerNumero(),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")

##Cargar el modelo .h5
model = tf.keras.models.load_model(
    # '/Users/asik/Desktop/ANPR/object_detection.h5'
    '/FLASH-User/Documents/10mo/TT_v2/Proy_TT/models/tl_4.h5'
    )