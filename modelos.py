# Red convolucional
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout,\
    MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow_addons as tfa
from sklearn.metrics import f1_score, accuracy_score, precision_score,\
    recall_score
from keras import backend as K

# Regresión logística
def regresion_logistica(X_train, X_test, y_train, y_test, C, 
                        regularization_methods, max_iterations):
 """
 Genera un modelo de regresión logística con los hiperparámetros que 
 se le hayan pasado en listas.
 
 Variables de entrada:
  X_train: Array X de entrenamiento
  X_test: Array X de validación
  y_train: Valores objetivo de entrenamiento
  y_test: Valores objetivo de validación
  C: Intensidad de la penalización
  regularization_methods: Regularización l1 (lasso) o l2 (Ridge)
  max_iterations: Número máximo de iteraciones
  
 Variable de salida:
  var_results_log: Diccionario con todos los modelos probados y
  su valor de accuracy, recall, precision y f1
 
 Ejemplo: var_result_log = regresion_logistica(X_train, X_test, y_train,
        y_test, C = [0.7,0.8,0.9], regularization_methods = ['l1','l2'],
        max_iterations = [100, 1000, 10000, 100000])
  

    var_results_log = []

    # Bucle for para probar diferentes combinaciones de hiperparámetros
    for c_value in C:
        for regularization_method in regularization_methods:
            for max_iteration in max_iterations:
                if regularization_method == 'l1':
                    log_reg = LogisticRegression(C=c_value,
                                         penalty='l1',
                                         solver='liblinear',
                                         max_iter=max_iteration,
                                         random_state =112)
                else:
                    log_reg = LogisticRegression(C=c_value,
                                         penalty='l2',
                                         max_iter=max_iteration,
                                         random_state =112)
            
             # Entrena el modelo con el conjunto de entrenamiento
            log_reg.fit(X_train, y_train)
            # Realiza predicciones con el conjunto de prueba
            y_pred = log_reg.predict(X_test)

            # Calculamos los estadísticos
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Almacena los resultados para comparar más tarde
            var_results_log.append({"C": c_value,
                            "regularization_method": regularization_method,
                            "max_iter": max_iteration,
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall:": recall,
                            "f1_score": f1})
    
    return var_results_log


# XGBOOST
def func_xgboost(X_train, X_test, y_train, y_test, learning_rate, max_depth,
                n_estimators):

    var_results_xgboost = []

    for lr in learning_rate:
        for md in max_depth:
            for n_est in n_estimators:
                model = xgb.XGBClassifier(objective='binary:logistic', 
                                      learning_rate = lr,
                                      max_depth = md,
                                      n_estimators=n_est,
                                      seed=112)

                # Entrenamos el modelo
                model.fit(X_train, y_train)

                # Predecimos los datos de test
                y_pred = model.predict(X_test)

                # Calculamos los estadísticos
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)

                # Almacena los resultados para comparar más tarde
                var_results_xgboost.append({"lr": lr,
                            "max_depth": md,
                            "n_estimators": n_est,
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall:": recall,
                            "f1_score": f1,
                            "confusion_matrix": cm})
    return var_results_xgboost
    
    # Red densa
def red_densa(train, test, lst_neuronas, lst_lr, lst_epoch, input_shape):
  var_results_dnn = []
  for neuronas in lst_neuronas:
      for lr in lst_lr:
          for epochs in lst_epoch:
              # Crear el modelo ANN
              model1 = Sequential()
              model1.add(Dense(neuronas, activation='relu', input_shape=(input_shape,)))
              model1.add(Dropout(0.2))
              model1.add(Dense(neuronas, activation='relu'))
              model1.add(Dropout(0.2))
              model1.add(Dense(1, activation='sigmoid'))

              # Compile
              loss_fn = keras.losses.BinaryCrossentropy()
              opt = keras.optimizers.Adam(learning_rate = lr)
              model1.compile(optimizer=opt, loss=loss_fn,
                              metrics=['accuracy',
                                      tfa.metrics.F1Score(
                                          num_classes=2,average="micro",
                                          threshold=0.75)])

              model_dense = model1.fit(train,
                                      epochs=epochs, 
                                      validation_data=test,
                                       verbose = 0)
          
              val_accuracy = model_dense.history["val_accuracy"][epochs-1]
              val_f1_score = model_dense.history["val_f1_score"][epochs-1]
              
              # Almacena los resultados para comparar más tarde
              var_results_dnn.append({"neuronas": neuronas,
                              "learning rate": lr,
                              "epochs": epochs,
                              "accuracy": val_accuracy,
                              "f1_score": val_f1_score})
              
              print("learning rate {0}, epochs {1} neuronas {2} accuracy {3} y f1_score {4}".format(
                lr, epochs, neuronas, val_accuracy, val_f1_score
            ))
  
  return var_results_dnn
  
def red_convolucional(train, test, kernel_size, lst_lr, lst_epoch, lado):
  print("Comienza el entrenamiento ... ")
  var_results_dnn = []

  for ks in kernel_size:
    for lr in lst_lr:
        for epochs in lst_epoch:
            # Crear el modelo ANN
            model2 = Sequential()
            model2.add(Conv2D(16, kernel_size=(ks, ks), activation='relu',
                              input_shape=(lado, lado, 1)))
            model2.add(Conv2D(32, (ks, ks), activation='relu', padding='same'))
            model2.add(MaxPooling2D(pool_size=(2, 2)))
            model2.add(Dropout(0.2))
            model2.add(Conv2D(64, (ks, ks), activation='relu', padding='same'))
            model2.add(Conv2D(64, (ks, ks), activation='relu', padding='same'))
            model2.add(MaxPooling2D(pool_size=(2, 2)))
            model2.add(Dropout(0.2))
            model2.add(Flatten())
            model2.add(Dense(1, activation='sigmoid'))

            # Compile
            loss_fn = keras.losses.BinaryCrossentropy()
            opt = keras.optimizers.Adam(learning_rate = lr)
            model2.compile(optimizer=opt, loss=loss_fn,
                            metrics=['accuracy',
                                    tfa.metrics.F1Score(
                                        num_classes=2,average="micro",
                                        threshold=0.75)])

            model_conv = model2.fit(train_conv,
                                    epochs=epochs, 
                                     validation_data=validation_conv,
                                    verbose = 0)
        
            val_accuracy = model_conv.history["val_accuracy"][epochs-1]
            val_f1_score = model_conv.history["val_f1_score"][epochs-1]
            
            # Almacena los resultados para comparar más tarde
            var_results_dnn.append({"learning rate": lr,
                            "epochs": epochs,
                            "kernel_size": ks,
                            "accuracy": val_accuracy,
                            "f1_score": val_f1_score})
            
            print("learning rate {0}, epochs {1} kernel size {2}, accuracy {3} y f1_score {4}".format(
                lr, epochs, ks, val_accuracy, val_f1_score
            ))
  return var_results_dnn


# Incorpora red convolucional con penalizaciones
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout,\
    MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow_addons as tfa
from sklearn.metrics import f1_score, accuracy_score, precision_score,\
    recall_score
from keras import backend as K
from keras.regularizers import l2

def red_convolucional_penal(train, test, kernel_size, lst_lr, lst_epoch, lado):
  print("Comienza el entrenamiento ... ")
  var_results_dnn = []
  regularization = l2(0.01)

  for ks in kernel_size:
    for lr in lst_lr:
        for epochs in lst_epoch:
            # Crear el modelo ANN
            model2 = Sequential()
            model2.add(Conv2D(16, kernel_size=(ks, ks), activation='relu',
                              input_shape=(lado, lado, 1),
                              kernel_regularizer=regularization))
            model2.add(Conv2D(32, (ks, ks), activation='relu', padding='same'))
            model2.add(MaxPooling2D(pool_size=(2, 2)))
            model2.add(Dropout(0.2))
            model2.add(Conv2D(64, (ks, ks), activation='relu', padding='same'))
            model2.add(Conv2D(64, (ks, ks), activation='relu', padding='same'))
            model2.add(MaxPooling2D(pool_size=(2, 2)))
            model2.add(Dropout(0.2))
            model2.add(Flatten())
            model2.add(Dense(1, activation='sigmoid'))

            # Compile
            loss_fn = keras.losses.BinaryCrossentropy()
            opt = keras.optimizers.Adam(learning_rate = lr)
            model2.compile(optimizer=opt, loss=loss_fn,
                            metrics=['accuracy',
                                    tfa.metrics.F1Score(
                                        num_classes=2,average="micro",
                                        threshold=0.75)])

            model_conv = model2.fit(train_conv,
                                    epochs=epochs, 
                                     validation_data=validation_conv)
        
            val_accuracy = model_conv.history["val_accuracy"][epochs-1]
            val_f1_score = model_conv.history["val_f1_score"][epochs-1]
            
            # Almacena los resultados para comparar más tarde
            var_results_dnn.append({"learning rate": lr,
                            "epochs": epochs,
                            "kernel_size": ks,
                            "accuracy": val_accuracy,
                            "f1_score": val_f1_score})
            
            print("learning rate {0}, epochs {1} kernel size {2}, accuracy {3} y f1_score {4}".format(
                lr, epochs, ks, val_accuracy, val_f1_score
            ))
  return var_results_dnn
