import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

#variables
kernel = np.ones((6, 6), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
orb = cv2.ORB_create()

def detect_can(contours):
    orientation = 0
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if (w > .5*h-.2*h and w < .5*h+.2*h):
            orientation = 1
        elif (w>2*h-.1*h and w<2*h+.1*h):
            orientation = 2
        if orientation > 0 and (cv2.contourArea(contours[i])) > 300:
            return i, orientation
        orientation = 0
    return -1, 0


# Iniciar camara
captura = cv2.VideoCapture(0)

while (1):
    _, img = captura.read()

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img = cv2.resize(hsv_img,(640,480))

    cv2.circle(img, (320, 240), 3, (0, 0, 255), -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Every color except white
    low = np.array([0, 42, 0])
    high = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_img, low, high)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Red color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv_img, low_red, high_red)
    red = cv2.bitwise_and(img, img, mask=red_mask)

    #green color
    green_low = np.array([49, 50, 50])
    green_high = np.array([100, 255, 210])
    mask_green = cv2.inRange(hsv_img, green_low, green_high)

    #blue color
    #blue_low = np.array([94, 80, 2])
    blue_low = np.array([49, 50, 50])
    #blue_high = np.array([126, 255, 255])
    blue_high = np.array([100, 255, 210])
    mask_blue = cv2.inRange(hsv_img, blue_low, blue_high)

    # Filtrar el ruido con un CLOSE seguido de un OPEN
    img_filtrate = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    img_filtrate = cv2.morphologyEx(img_filtrate, cv2.MORPH_OPEN, kernel)

    # Difuminamos la mascara para suavizar los contornos y aplicamos filtro canny
    #blur = cv2.Canny(img_filtrate, 1, 2)
    _, Bin =cv2.threshold(img_filtrate,165,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    blur = cv2.GaussianBlur(img_filtrate, (5, 5), 0)
    #blur = cv2.Canny(blur, 1, 2)

    #buscamos contornos
    contours_green, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    index, orientation = detect_can(contours_blue)
    print(index,'',orientation)

    for c in contours_blue:
        area=cv2.contourArea(c)
        if (area >2000) & (area < 150000) and index > -1:
            momentos = cv2.moments(c)
            cx = int(momentos['m10'] / momentos['m00'])
            cy = int(momentos['m01'] / momentos['m00'])
            cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(img, "(x: " + str(cx) + ", y: " + str(cy) + ")", (cx + 10, cy + 10), font, 0.5,
                        (255, 255, 255), 1)

            nuevoContorno = cv2.convexHull(c)
            new_image = cv2.drawContours(img, [nuevoContorno], -1, (0, 0, 255), 1)

            x, y, w, h = cv2.boundingRect(nuevoContorno)
            rectangulo = cv2.rectangle(img, (x-25, y-25), (x + w + 25, y + h + 25), (0, 255, 0), 1)
            final = cv2.bitwise_and(img, img, mask=blur)
            kp = orb.detect(final, None)
            #kp, des = orb.compute(final, kp)
            #draw only keypoints location,not size and orientation
            #img2 = cv2.drawKeypoints(img, kp, img, flags=0)
            #cv2.imshow('final', final)
            huMoments = cv2.HuMoments(momentos)
            #cv2.imshow('final',final)
            #print((x+w+25)/(y+h+25))
            #plt.imshow(huMoments)


    cv2.imshow('mask', blur)
    cv2.imshow('img', img)
    cv2.imshow('frame', Bin)
    #plt.show()

    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break

def realce(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    print(gray)


cv2.destroyAllWindows()