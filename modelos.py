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
