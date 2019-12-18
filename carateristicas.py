import numpy as np
import cv2 as cv
import math as mt

def get_img_lata(i):                         #funcion de captura de imagen
    file = 'C:/Users/mag_g/Documents/uv/TOPICOS DE IA/codigo prueba de robot/train data/(' + str(i) +').JPG'       #ruta de imagen
    img = cv.imread(file)               #lectura de imagen
    img = cv.resize(img,(640,480))      #redefinir imagen
    return img

def get_img_sin_lata(i):                         #funcion de captura de imagen
    file = 'C:/Users/mag_g/Documents/uv/TOPICOS DE IA/codigo prueba de robot/objetos verdes/ (' + str(i) +').jpg'       #ruta de imagen
    img = cv.imread(file)               #lectura de imagen
    img = cv.resize(img,(640,480))      #redefinir imagen
    return img

def preprocesamiento_img(img):          #funcion de procesamiento de imagen
    kernel = np.ones((3, 3), np.uint8)  #filtros para erocion o dilatacion

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)    #convercion de bgr a hsv
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)      #convercion a escale de grises

### umbrales para la detencion de color verde
    blue_low = np.array([49, 50, 50])
    blue_high = np.array([100, 255, 210])
    mask_blue = cv.inRange(hsv_img, blue_low, blue_high)

### eliminacion de ruido mediante filtro morfologico
    mask_blue = cv.morphologyEx(mask_blue, cv.MORPH_CLOSE, kernel)
    mask_blue = cv.morphologyEx(mask_blue, cv.MORPH_OPEN, kernel)

    mask_blue = cv.medianBlur(mask_blue,3) #blur a imagen para quitar efecto de pimienta y sal
    contours_blue, _ = cv.findContours(mask_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #busqueda de contornos

    for c in contours_blue:
        area = cv.contourArea(c) #area del contorno encontrado
        if area > 1000: #eliminacion de areas pequeñas
            nuevoContorno = cv.convexHull(c) #alisado de contorno
            cv.drawContours(img, [nuevoContorno], -1, (0, 0, 255), 1) #dibuja los contornos en la imagen original
            x, y, w, h = cv.boundingRect(nuevoContorno) #encientra regiones rectangulares y regresa puntos de esquinas
            cv.rectangle(img, (x - 25,   y - 25), (x + w + 25, y + h + 25), (0, 0, 255), 3) #dibujado de momentos de la deseada
            momentos = cv.moments(nuevoContorno) #se calculan los momentos
            return momentos, mask_blue #regresa el valor si se encontro con una region de interes

def get_humoments(momentos):
    hu = []
    huMoments = cv.HuMoments(momentos) #calculo de los momentos de hu
    for i in range(0, 7):
        hu.append(huMoments[i][0])      #gregar los valores a un array

    for i in range(0, 7):
        hu[i] = -1 * mt.copysign(1.0, hu[i]) * mt.log10(abs(hu[i])) # convertir de escala logaritmica o exponencial
    return hu #regresa los momentos de hu

### creacion del archivo para almacenar las caracteristicas con lata
data = {}
archivo = 'caracteristicas con lata.csv'
csv = open(archivo, 'w')
columnas = 'h0,h1,h2,h3,h4,h5,h6,h7\n'
csv.write(columnas)

for j in range(1,50):
    img = get_img_lata(j)    #lee el conjunto de imagenes de entrenamieno
    if img is None:
        print("esta imagen no esta"+str(j))
    else:
        #se almacenan los datos en un diccionario y despues se pasa a un archivo csv
        momentos, mask_blue = preprocesamiento_img(img)
        cv.imshow('img', mask_blue)
        cv.waitKey(2)
        huMoments = get_humoments(momentos)
        print('imagen: ',j,' ',huMoments)
        for i in range(0,7):
            data['h' + str(i+1)] = str(huMoments[i])
        for key in data.keys():
            h = data[key]
            fila = h + ','
            csv.write(fila)
        csv.write('\n')
        huMoments = 0

### creacion del archivo para almacenar las caracteristicas sin lata

archivo2 = 'caracteristicas sin lata.csv'
csv2 = open(archivo2, 'w')
columnas2 = 'h0,h1,h2,h3,h4,h5,h6,h7\n'
csv2.write(columnas2)
for j in range(1,50):
    img = get_img_sin_lata(j)    #lee el conjunto de imagenes de entrenamieno
    if img is None:
        print("esta imagen no esta"+str(j))
    else:
        #se almacenan los datos en un diccionario y despues se pasa a un archivo csv
        momentos, mask_blue = preprocesamiento_img(img)
        cv.imshow('img', mask_blue)
        cv.waitKey(2)
        huMoments = get_humoments(momentos)
        print('imagen: ',j,' ',huMoments)
        for i in range(0,7):
            data['h' + str(i+1)] = str(huMoments[i])
        for key in data.keys():
            h = data[key]
            fila = h + ','
            csv2.write(fila)
        csv2.write('\n')
        huMoments = 0

cv.destroyAllWindows()