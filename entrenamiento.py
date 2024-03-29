import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import cv2

def genera_dataframe(ruta_incendio, ruta_no_incendio, radar, reshape_x,
                     reshape_y):
    
  """
  Esta función parte de las rutas en la que se les haya marcado incendios y no
  incendios y devuelve un dataframe con el ancho solicitado.

  Variables entrada:
    ruta_incendio: Carpeta donde se encuentran las imágenes de incendio
    ruta_no_incendio: Carpeta donde se encuentran las imágenes de no incendio
    radar: tipo de radar (LST_Day, NDVI, et...)
    reshape_x: número de píxeles de ancho que se espera.
    reshape_y: número de píxeles de alto que se espera.

  Ejemplo de llamada:
    df_temperatura = genera_dataframe(ruta_incendio = path_incendios,
                                      ruta_no_incendio = path_no_incendios,
                                      radar= "LST_Day",
                                      reshape_x = 23,
                                      reshape_y = 23)
  """
  lst_archivos_incendios = []
  lst_archivos_no_incendios = []
  # Listado de incendios
  lista_ficheros_incendios = []
  n_varibles_x =reshape_x * reshape_y
  print("Creando lista de incendios a leer...")
  for file in os.listdir(ruta_incendio):
      if radar in file and 'jpeg' in file:
            lst_archivos_incendios.append(file)

  for file in os.listdir(ruta_no_incendio):
      if radar in file and 'jpeg' in file:
          lst_archivos_no_incendios.append(file)  
  print("Leyendo imágenes...")

  contador = 0
    
  for file in  lst_archivos_incendios:
      imagen = cv2.imread(ruta_incendio + file)[0:reshape_x, 0:reshape_y, :]
      image_grey = np.array(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY))
      if "incendio" in file:
        image_grey = cv2.resize(image_grey, (0,0), fx=0.01, fy=0.01) # Reduzcol 99%
        n_varibles_x =int(reshape_x/100 * reshape_y/100)
      image_grey_reshape = image_grey.reshape(n_varibles_x)
      lst_grey_reshape = image_grey_reshape.tolist()
      lst_grey_reshape.append(1) # Incorporo el target
      lista_ficheros_incendios.append(lst_grey_reshape)
      contador = contador + 1
      print("Imágenes leídas: {}".format(contador), end='\r')

  for file in  lst_archivos_no_incendios:
      imagen = cv2.imread(ruta_no_incendio + file)[0:reshape_x, 0:reshape_y, :]
      image_grey = np.array(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY))
      if "incendio" in file:
        image_grey = cv2.resize(image_grey, (0,0), fx=0.01, fy=0.01) # Reduzcol 99%
        n_varibles_x =int(reshape_x/100 * reshape_y/100)
      image_grey_reshape = image_grey.reshape(n_varibles_x)
      lst_grey_reshape = image_grey_reshape.tolist()
      lst_grey_reshape.append(0) # Incorporo el target
      lista_ficheros_incendios.append(lst_grey_reshape)
      contador = contador + 1
      print("Imágenes leídas: {}".format(contador), end='\r')
    
  # Lo paso a DataFrame
  print("Lo paso a dataframe...")
  lst_features = []
  lst_features_x = []
  for i in range(n_varibles_x):
      lst_features.append("col" + str(i))
      lst_features_x.append("col" + str(i))
  lst_features.append("target")
  # Genero el dataframe en sí
  df = pd.DataFrame(data = lista_ficheros_incendios, columns =lst_features )

  lista_id = []
  for item in lst_archivos_incendios:
    id = int(item.split(radar[0])[0])
    lista_id.append(id)
  for item in lst_archivos_no_incendios:
    id = int(item.split(radar[0])[0])
    lista_id.append(id)

  print(len(lst_archivos_incendios), len(lst_archivos_no_incendios), len(lista_id),
        len(df))

  df_id = pd.DataFrame({'id': lista_id})
  df = pd.concat([df, df_id], axis=1)
  print(len(df))
  # Barajo el dataframe
  df=df.sample(frac=1).reset_index(drop=True)

  return df, lst_features_x, df_id


def almacena_bn(ruta_incendio, ruta_no_incendio, radar):
  """
  Esta función está especialmente creada para convertir a blanco y negro las
  imágenes reales

  Variables entrada:
    ruta_incendio: Carpeta donde se encuentran las imágenes de incendio
    ruta_no_incendio: Carpeta donde se encuentran las imágenes de no incendio
    radar: tipo de imagen

  Ejemplo de llamada:
    df_temperatura = genera_dataframe(ruta_incendio = path_incendios,
                                      ruta_no_incendio = path_no_incendios,
                                      radar= "incendios")
  """
  # Listado de incendios
  lista_ficheros_incendios = []
  for file in os.listdir(path_incendios):
    if radar in file:
      imagen = cv2.imread(path_incendios + file)
      image_grey = np.array(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY))
      cv2.imwrite(path_incendios + 'g' + file, image_grey)

  # Listado de no incendios
  lista_ficheros_no_incendios = []
  for file in os.listdir(path_no_incendios):
    if radar in file:
      imagen = cv2.imread(path_no_incendios + file)
      image_grey = np.array(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY))
      cv2.imwrite(path_no_incendios + 'g' + file, image_grey)


def genera_dataframe_bn(ruta_incendio, ruta_no_incendio, radar, extension,
                        reshape_x, reshape_y):
  """
  Esta función parte de las rutas en la que se les haya marcado incendios y no
  incendios y devuelve un dataframe con el ancho solicitado.

  Variables entrada:
    ruta_incendio: Carpeta donde se encuentran las imágenes de incendio
    ruta_no_incendio: Carpeta donde se encuentran las imágenes de no incendio
    radar: tipo de radar (LST_Day, NDVI, et...)
    reshape_x: número de píxeles de ancho que se espera.
    reshape_y: número de píxeles de alto que se espera.

  Ejemplo de llamada:
    df_temperatura = genera_dataframe(ruta_incendio = path_incendios,
                                      ruta_no_incendio = path_no_incendios,
                                      radar= "LST_Day",
                                      reshape_x = 23,
                                      reshape_y = 23)
  """
  # Listado de incendios
  lista_ficheros_incendios = []
  count = 0
  n_varibles_x =reshape_x * reshape_y
  for file in os.listdir(path_incendios):
    print("imagen: {}".format(path_incendios + file))
    if (radar in file) and (extension in file):
      imagen = cv2.imread(path_incendios + file)[0:reshape_x, 0:reshape_y, :]
      image_grey = np.array(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY))
      image_grey_reshape = image_grey.reshape(n_varibles_x)
      lst_grey_reshape = image_grey_reshape.tolist()
      lst_grey_reshape.append(1) # Incorporo el target
      lista_ficheros_incendios.append(lst_grey_reshape)
    #count = count + 1
    #print(count, end = '\r')

  # Listado de no incendios
  lista_ficheros_no_incendios = []
  for file in os.listdir(path_no_incendios):
    if (radar in file) and (extension in file):
      imagen = cv2.imread(path_no_incendios + file)[0:reshape_x, 0:reshape_y, :]
      image_grey = np.array(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY))
      image_grey_reshape = image_grey.reshape(n_varibles_x)
      lst_grey_reshape = image_grey_reshape.tolist()
      lst_grey_reshape.append(0) # Incorporo el target
      lista_ficheros_incendios.append(lst_grey_reshape)
    #count = count + 1
    #print(count, end = '\r')

  # Lo paso a DataFrame
  lst_features = []
  lst_features_x = []
  for i in range(n_varibles_x):
    lst_features.append("col" + str(i))
    lst_features_x.append("col" + str(i))
  lst_features.append("target")

  # Genero el dataframe en sí
  df = pd.DataFrame(data = lista_ficheros_incendios, columns =lst_features )

  # Barajo el dataframe
  df=df.sample(frac=1).reset_index(drop=True)

  return df

def borrado_entrenamiento_test():
  """
  La siguiente función borra todas las imágenes en las direcciones principales de guardado
  de imágenes
  """
  path_train_incendio = '/content/gdrive/My Drive/incendios_satelite/Entrenamiento_incendio/'
  path_train_no_incendio = '/content/gdrive/My Drive/incendios_satelite/Entrenamiento_no_incendio/'
  path_test_incendio = '/content/gdrive/My Drive/incendios_satelite/Validacion_incendio/'
  path_test_no_incendio = '/content/gdrive/My Drive/incendios_satelite/Validacion_no_incendio/'
  # Quito las imágenes negras o las que tengan bordes negros
  for file in os.listdir(path_train_incendio):
    os.remove(path_train_incendio + file)
  for file in os.listdir(path_train_no_incendio):
    os.remove(path_train_no_incendio + file)
  for file in os.listdir(path_test_incendio):
    os.remove(path_test_incendio + file)
  for file in os.listdir(path_test_no_incendio):
    os.remove(path_test_no_incendio + file)

# Plot del training loss y el accuracy
def plot_prediction(n_epochs, mfit):
  """
  Esta función dibuja la función de pérdida y el valor de f1 tanto del
  entrenamiento como de la validación.
  Variables de entrada:
    n_epochs: Número de pasos que se haya utilizado en la Red
    mfit: Modelo entrenado
  
  Ejemplo:
    plot_prediction(n_epochs = 200, mfit = mod_conv):
  """
    plt.plot(mfit.history['loss'])
    plt.plot(mfit.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    plt.plot(mfit.history['accuracy'])
    plt.plot(mfit.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    plt.plot(mfit.history['f1_score'])
    plt.plot(mfit.history['val_f1_score'])
    plt.ylabel('f1_score')
    plt.xlabel('epoch')

def convertir_jpeg(path):
  """
  La siguiente función convierte todas las imágenes de una ruta a formato
  JPEG que es el que lee el data loading utilizado para las redes neuronales
  artificiales

  Variable de entrada:
    Path: Ruta de google drive donde va a leer las imágenes

  Ejemplo:
  convertir_jpeg('/content/gdrive/My Drive/incendios_satelite/Incendios masivos/')
  """
    for file in os.listdir(path):
        file_no_ext = file[0:-4]
        im = Image.open(path + file)
        im.thumbnail(im.size)
        im.save(path + file_no_ext + ".jpeg", "JPEG", quality=100)

def preprocesado(df):
  """
  Preprocesamiento para los modelos de ML. Separa la muestra de train y test de
  X e y con 20%-80% y estandariza todas las variables X.
  
  Variable de entrada:
    df: Dataframe que se quiere preprocesar
    
   Variable de salida:
    X_train: Variables predictoras de entrenamiento
    X_test: Variables predictoras de validación
    y_train: Variable objetivo de entrenamiento
    y_test: Variable objetivo de validación
    
  Ejemplo:
    X_train, X_test, y_train, y_test = preprocesado(df = df_temperatura)
  """
    # Procesos previos
    # Separamos los datos
    X = df[lst_features_x]
    y = df["target"]

    # Dividimos los datos en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
  
def preprocesado(df_train, df_test):
  """
  Preprocesamiento para los modelos de DL. Separa la muestra de train y test de
  X e y utilizando los dataframe que se le están pasando como parámetros.
  
  Variable de entrada:
    df_train: Dataframe de entrenamiento
    df_test: Dataframe de validacion
    
   Variable de salida:
    X_train: Variables predictoras de entrenamiento
    X_test: Variables predictoras de validación
    y_train: Variable objetivo de entrenamiento
    y_test: Variable objetivo de validación
    
  Ejemplo:
    X_train, X_test, y_train, y_test = preprocesado(df_train = df_temperatura_train
                                                    df_test = df_temperatura_test)
  """
    # Procesos previos
    # Separamos los datos
    X_train = df_train[lst_features_x]
    y_train = df_train["target"]
    X_test = df_test[lst_features_x]
    y_test = df_test["target"]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
