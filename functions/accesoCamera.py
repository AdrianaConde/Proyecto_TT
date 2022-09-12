#Deteccion en videos
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# salida = cv2.VideoWriter('Camera.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (640,480)) 
with mp_hands.Hands(
    static_image_mode=False, #Video
    # static_image_mode=True, #Imagenes 
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:

    while cap.isOpened(): #True: 
        ret, frame = cap.read()
        # salida.write(frame)
        if ret == False:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks is not None: #para puntos de referencia de la mano
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=3, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=1))
        else:
            font = cv2.FONT_HERSHEY_COMPLEX ##FONT_HERSHEY_SIMPLEX 
            cv2.putText(
                frame, 'NO SE DETECTA LA MANO', (50, 50), font, 1.3, (255, 0, 204 ), 2, cv2.LINE_8)         
            # cv2.imshow('-------Access to camera-------',frame)
        if (cv2.waitKey(1) == ord('s')): #s salir del blucle
            break
# salida.release()   
cap.release()
# cv2.destroyAllWindows()
