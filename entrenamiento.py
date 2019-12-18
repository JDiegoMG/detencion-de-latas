import cv2 as cv
import pandas as pd
import numpy as np
import math as mt
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer
import pickle

#lectura de los datos con las caracteteristicas de cada imagen
data = []
archivo = 'caracteristicas con lata.csv'
datos =  pd.read_csv(archivo, header=0)

data2 = []
archivo2 = 'caracteristicas sin lata.csv'
datos2 =  pd.read_csv(archivo2, header=0)

#configuracion de la red neuronal (caracteristicas, capas de la red, salidas)
net = buildNetwork(7, 40, 2, bias=True)
ds = SupervisedDataSet(7, 2)

#lecctura de cada dato de la imagen
for j in range(0,len(datos)):
    data.clear()
    data2.clear()
    for i in range(0,7):
        data.append(datos.iloc[j,i])
        data2.append(datos2.iloc[j,i])
    #print('paso: ', data)
    ds.addSample((data), (1,))  #asignacion de caracteristicas y etiqueta
    ds.addSample((data2), (0,))
    #print('paso: ', j)
    #print(ds)
    trainer = BackpropTrainer(net, ds) #entrenamiento de la red mediante la comparacion del error esperado
    er = round(trainer.train(), 3)
    while er > 0.112: #error esperado
        er = round(trainer.train(), 3)
        #print(er)

filename = "NNa{0}b{1}.pk1".format(2,3)
pickle.dump(net, open(filename,'wb'))

def preprocesamiento_img(img):
    kernel = np.ones((3, 3), np.uint8)
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist = cv.equalizeHist(gray, dst=None)

    blue_low = np.array([49, 50, 50])
    blue_high = np.array([100, 255, 210])
    mask_blue = cv.inRange(hsv_img, blue_low, blue_high)

    mask_blue = cv.morphologyEx(mask_blue, cv.MORPH_CLOSE, kernel)
    mask_blue = cv.morphologyEx(mask_blue, cv.MORPH_OPEN, kernel)

    mask_blue = cv.medianBlur(mask_blue,3)
    contours_blue, _ = cv.findContours(mask_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for c in contours_blue:
        area = cv.contourArea(c)
        if area > 1000:
            nuevoContorno = cv.convexHull(c)
            cv.drawContours(img, [nuevoContorno], -1, (0, 255, 0), 1)
            x, y, w, h = cv.boundingRect(nuevoContorno)
            rec = cv.rectangle(img, (x - 25, y - 25), (x + w + 25, y + h + 25), (0, 0, 255), 3)
            momentos = cv.moments(nuevoContorno)
            #cv.imshow('img', img)
            #cv.waitKey()
            return momentos

def get_humoments(momentos):
    hu = []
    huMoments = cv.HuMoments(momentos)
    for i in range(0, 7):
        hu.append(huMoments[i][0])

    for i in range(0, 7):
        hu[i] = -1 * mt.copysign(1.0, hu[i]) * mt.log10(abs(hu[i]))
    return hu

#lectura de la imagen de prediccion
suma = 0
suma2 = 0
suma3 = 0
clase_predicha = []
for i in range(1,21):
    suma += 1
    file = 'C:/Users/mag_g/Documents/uv/TOPICOS DE IA/detencion de latas/imagen test/ ('+ str(i) + ').jpg'
    img = cv.imread(file)
    img = cv.resize(img,(640,480))
    momentos = preprocesamiento_img(img)
    #cv.imshow('Imagen de prediccion',img)
    #cv.waitKey()
    hu = get_humoments(momentos)
    compute = net.activate(hu) #presiccion con el modelo entrenado
    r = round(compute[0], 0)
    clase_predicha.append(abs(r))
    if r==1:
        print('es una lata')
        suma2 += 1

    if r==0:
        print('no es lata')
        suma3 += 1

etiquetas_test = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
mc = confusion_matrix(etiquetas_test, clase_predicha)
print (u'Matriz de confusión')
print (mc)

plt.figure(figsize=(6,6))
ticks = range(10)
plt.xticks(ticks)
plt.yticks(ticks)
plt.imshow(mc,cmap=plt.cm.Blues,interpolation='nearest')
plt.colorbar(shrink=0.8)
w, h = mc.shape
for i in range(w):
    for j in range(h):
        plt.annotate(str(mc[i][j]), xy=(j, i),
                    horizontalalignment='center',
                    verticalalignment='center')
plt.xlabel('Etiqueta predicha')
plt.ylabel('Etiqueta')
plt.title(u'Matriz de Confusión')
plt.show()

cv.destroyAllWindows()

cv.destroyAllWindows()

