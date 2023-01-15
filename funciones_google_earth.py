# Functions
# Generamos una máscara de nubes para seleccionar como de cubierto debe estar
def maskS2clouds(image):
  """
  Esta función prepara el mapa de nubes para tratarlas en la extracción del
  mapa.

  Entrada de variables:
    image: Nombra la variable sobre la que retornará la selección

  Salida: Máscara de imágenes

  Ejemplo de llamada:
  imagen = imagen.map(maskS2clouds)
  """
  qa = image.select('QA60')

  # La máscara 10 y 11 hacen referencia a las nuebes
  cloudBitMask = 1 << 10
  cirrusBitMask = 1 << 11

  # El 0 indica que se encuentra despejado
  mask = qa.bitwiseAnd(cloudBitMask).eq(0)
  mask = mask.bitwiseAnd(cirrusBitMask).eq(0)

  return image.updateMask(mask).divide(10000)
  

def incendio(id, x, y, year, month, day, anticipacion):
  """
  En función de las coordenas y la fecha seleccionada, lanza una consulta a
  google earth engine para que descargue una imagen real del tamaño de 0.2 tanto
  en lpongitud como en latitud en decimal

    Variables de entrada:
      id: id de la imagen
      x: Coordenadas en x de coordenadas decimales
      y: Coordenadas en y de coordenadas decimales
      year: Año de selección de imagen
      month: Mes de de selección de imagen
      day: Día de selección de imagen
      anticipación: Días anteriores a la fecha del evento que queremos estudiar

    Ejemplo de llamada:
    df_satelite_control.apply(lambda x: incendio(id = x["id"],
                                     x = x["X"],
                                     y = x["Y"],
                                     year = x["year"],
                                     month = x["month"],
                                     day = x["day"],
                                     anticipacion = 1), axis=1)
  """
  fecha_calculo = datetime.date(int(year), int(month), int(day)) -\
     datetime.timedelta(days=anticipacion)
  fecha_inicio = fecha_calculo - datetime.timedelta(days=1)
  fecha_fin = fecha_inicio + datetime.timedelta(days=1)
  id = id
  longitud = x
  latitud = y

  geom = ee.Geometry.Polygon([[latitud-0.1, longitud-0.1],
                              [latitud-0.1, longitud+0.1],
                              [latitud+0.1, longitud-0.1],
                              [latitud+0.1, longitud+0.1]])

  # Llamamos a una colección de imágenes aunque luego nos quedaremos con sólo
  # una de ellas
  collection = (ee.ImageCollection("COPERNICUS/S2")
              # Seleccionamos los tres canales RGB y la marca  de nubes
                .select(['B4', 'B3', 'B2', 'QA60'])
              # Filter for images within a given date range.
                .filter(ee.Filter.date(fecha_inicio.strftime("%Y-%m-%d"), 
                                       fecha_fin.strftime("%Y-%m-%d")))
              # Seleccionamos las imágenes dentro del rectángulo que hemos hecho
                .filterBounds(geom)
              # Nos quedamos con las nubes con porcentaje inferior al 10%
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
              # Aplicamos máscara de nubes
                .map(maskS2clouds)
              )

    # Convierte la colección en una única imagen
  image = collection.sort('system:index', opt_ascending=False).mosaic()

  # Parámetros de la imagen que queremos observar
  image = image.visualize(bands=['B4', 'B3', 'B2'],
                          min=[0.0, 0.0, 0.0],
                          max=[0.3, 0.3, 0.3]
                        )
  

  fecha_inicio_str =  fecha_inicio.strftime("%Y-%m-%d")
  nombre_imagen = str(int(id))+"incendio" + fecha_inicio_str
  print(nombre_imagen)

    # Assign export parameters.
  task_config = {
      'region': geom.coordinates().getInfo(),
      'folder': 'incendios_satelite',
      'scale': 10,
      'crs': 'EPSG:4326',
      'description': nombre_imagen
    }

  # Export Image
  task = ee.batch.Export.image.toDrive(image, **task_config)
  task.start()


def temperatura(id, x, y, year, month, day, anticipacion):
  """
  En función de las coordenas y la fecha seleccionada, lanza una consulta a
  google earth engine para que descargue una imagen sobre la temperatura de
  la superficie del tamaño de 0.2 tanto   en lpongitud como en latitud en 
  decimal

    Variables de entrada:
      id: id de la imagen
      x: Coordenadas en x de coordenadas decimales
      y: Coordenadas en y de coordenadas decimales
      year: Año de selección de imagen
      month: Mes de de selección de imagen
      day: Día de selección de imagen
      anticipación: Días anteriores a la fecha del evento que queremos estudiar

  Ejemplo de llamada:
    df_satelite_control.apply(lambda x: temperatura(id = x["id"],
                                     x = x["X"],
                                     y = x["Y"],
                                     year = x["year"],
                                     month = x["month"],
                                     day = x["day"],
                                     anticipacion = 1), axis=1)
  """

  fecha_calculo = datetime.date(int(year), int(month), int(day)) -\
     datetime.timedelta(days=anticipacion)
  fecha_inicio = fecha_calculo - datetime.timedelta(days=1)
  fecha_fin = fecha_inicio + datetime.timedelta(days=8)
  id = id
  longitud = x
  latitud = y

  geom = ee.Geometry.Polygon([[latitud-0.1, longitud-0.1],
                              [latitud-0.1, longitud+0.1],
                              [latitud+0.1, longitud-0.1],
                              [latitud+0.1, longitud+0.1]])

  # Llamamos a una colección de imágenes aunque luego nos quedaremos con sólo una de ellas
  # Importar datos de google earth engine
  collection = ee.ImageCollection('MODIS/061/MOD11A1')

  # Seleccionar el periodo de tiempo
  collection = collection.filter(ee.Filter.date(fecha_inicio.strftime("%Y-%m-%d"), 
                                                fecha_fin.strftime("%Y-%m-%d")))

  # Seleccionar la banda de interes
  collection = collection.select('LST_Day_1km')

  collection=collection.filterBounds(geom)
  image = collection.mean()
    
  fecha_inicio_str =  fecha_inicio.strftime("%Y-%m-%d")
  nombre_imagen = str(int(id))+"LST_Day_1km" + fecha_inicio_str
    
  image = image.visualize(bands=['LST_Day_1km'],
                min=13000,
                max=16500,
                palette= ["040274","040281","0502a3","0502b8","0502ce","0502e6",
                          "0602ff","235cb1","307ef3","269db1","30c8e2","32d3ef",
                          "3be285","3ff38f","86e26f","3ae237","b5e22e","d6e21f",
                          "fff705","ffd611","ffb613","ff8b13","ff6e08","ff500d",
                          "ff0000","de0101","c21301","a71001","911003"])


  # Assign export parameters.
  task_config = {
      'region': geom.coordinates().getInfo(),
      'folder': 'incendios_satelite',
      'scale': 1000,
      'crs': 'EPSG:4326',
      'description': nombre_imagen
  }

    # Export Image
  task = ee.batch.Export.image.toDrive(image, **task_config)
  task.start()

def humedad_relativa(id, x, y, year, month, day, anticipacion):
  """
  En función de las coordenas y la fecha seleccionada, lanza una consulta a
  google earth engine para que descargue una imagen sobre la humedad relativa
  del fuel del tamaño de 0.2 tanto   en lpongitud como en latitud en decimal

    Variables de entrada:
      id: id de la imagen
      x: Coordenadas en x de coordenadas decimales
      y: Coordenadas en y de coordenadas decimales
      year: Año de selección de imagen
      month: Mes de de selección de imagen
      day: Día de selección de imagen
      anticipación: Días anteriores a la fecha del evento que queremos estudiar

  Ejemplo de llamada:
    df_satelite_control.apply(lambda x: humedad_relativa(id = x["id"],
                                     x = x["X"],
                                     y = x["Y"],
                                     year = x["year"],
                                     month = x["month"],
                                     day = x["day"],
                                     anticipacion = 1), axis=1)
  """
  fecha_calculo = datetime.date(int(year), int(month), int(day)) -\
       datetime.timedelta(days=anticipacion)
  fecha_inicio = fecha_calculo - datetime.timedelta(days=20)
  fecha_fin = fecha_inicio + datetime.timedelta(days=20)
  id = id
  longitud = x
  latitud = y
  geom = ee.Geometry.Polygon([[latitud-0.1, longitud-0.1],
                              [latitud-0.1, longitud+0.1],
                              [latitud+0.1, longitud-0.1],
                              [latitud+0.1, longitud+0.1]])


  # Importar datos de google earth engine
  dataset = ee.ImageCollection('MODIS/061/MOD13A2')

  # Seleccionar el periodo de tiempo
  dataset = dataset.filter(ee.Filter.date(fecha_inicio.strftime("%Y-%m-%d"), 
                                          fecha_fin.strftime("%Y-%m-%d")))

  # Seleccionar la banda de interes
  dataset = dataset.select('NDVI')

  dataset=dataset.filterBounds(geom)
  image = dataset.mean()

  fecha_inicio_str =  fecha_inicio.strftime("%Y-%m-%d")
  nombre_imagen = str(int(id)) + "NDVI" + fecha_inicio_str
  
  print(nombre_imagen)
  
  image = image.visualize(bands=['NDVI'],
           min=0,
           max=9000,
           palette= ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718',
                     '74A901','66A000', '529400', '3E8601', '207401', '056201',
                     '004C00', '023B01','012E01', '011D01', '011301'])


  # Assign export parameters.
  task_config = {
      'region': geom.coordinates().getInfo(),
      'folder': 'incendios_satelite',
      'scale': 1000,
      'crs': 'EPSG:4326',
      'description': nombre_imagen
    }

  # Export Image
  task = ee.batch.Export.image.toDrive(image, **task_config)
  task.start()


# La poblacion
def poblacion(id, x, y, year, month, day, anticipacion):
  """
  En función de las coordenas y la fecha seleccionada, lanza una consulta a
  google earth engine para que descargue una imagen sobre la densidad de 
  población del tamaño de 0.2 tanto   en lpongitud como en latitud en decimal

    Variables de entrada:
      id: id de la imagen
      x: Coordenadas en x de coordenadas decimales
      y: Coordenadas en y de coordenadas decimales
      year: Año de selección de imagen
      month: Mes de de selección de imagen
      day: Día de selección de imagen
      anticipación: Días anteriores a la fecha del evento que queremos estudiar

  Ejemplo de llamada:
    df_satelite_control.apply(lambda x: poblacion(id = x["id"],
                                     x = x["X"],
                                     y = x["Y"],
                                     year = x["year"],
                                     month = x["month"],
                                     day = x["day"],
                                     anticipacion = 1), axis=1)
  """
  fecha_calculo = datetime.date(2015,1,1)
  fecha_inicio = fecha_calculo
  fecha_fin = datetime.date(2022,6,30)
  id = id
  longitud = x
  latitud = y

  geom = ee.Geometry.Polygon([[latitud-0.1, longitud-0.1],
                              [latitud-0.1, longitud+0.1],
                              [latitud+0.1, longitud-0.1],
                              [latitud+0.1, longitud+0.1]])


  # Llamamos a una colección de imágenes aunque luego nos quedaremos con sólo una de ellas
  # Importar datos de google earth engine
  dataset = ee.ImageCollection('JRC/GHSL/P2016/SMOD_POP_GLOBE_V1')

  # Seleccionar el periodo de tiempo
  dataset = dataset.filter(ee.Filter.date(fecha_inicio.strftime("%Y-%m-%d"), 
                                          fecha_fin.strftime("%Y-%m-%d")))

  # Seleccionar la banda de interes
  dataset = dataset.select('smod_code')

  dataset=dataset.filterBounds(geom)
  image = dataset.mean()
  

  fecha_inicio_str =  fecha_inicio.strftime("%Y-%m-%d")
  nombre_imagen = str(int(id)) + "smod_code" + fecha_inicio_str
  
  print(nombre_imagen)
    
  image = image.visualize(bands=['smod_code'],
                          min=0,
                          max=3,
                          palette= ['000000', '448564', '70daa4', 'ffffff'])


  # Assign export parameters.
  task_config = {
      'region': geom.coordinates().getInfo(),
      'folder': 'incendios_satelite',
      'scale': 1000,
      'crs': 'EPSG:4326',
      'description': nombre_imagen
    }

    # Export Image
  task = ee.batch.Export.image.toDrive(image, **task_config)
  task.start()

# Viento
def viento(id, x, y, year, month, day, anticipacion, direccion):
  """
  En función de las coordenas y la fecha seleccionada, lanza una consulta a
  google earth engine para que descargue una imagen sobre la velocidad del
  viento en una orientación del tamaño de 0.2 tanto en lpongitud como en latitud
  en decimal

    Variables de entrada:
      id: id de la imagen
      x: Coordenadas en x de coordenadas decimales
      y: Coordenadas en y de coordenadas decimales
      year: Año de selección de imagen
      month: Mes de de selección de imagen
      day: Día de selección de imagen
      anticipación: Días anteriores a la fecha del evento que queremos estudiar
      dirección: norte "U" o este "V"

  Ejemplo de llamada:
    df_satelite_control.apply(lambda x: viento(id = x["id"],
                                     x = x["X"],
                                     y = x["Y"],
                                     year = x["year"],
                                     month = x["month"],
                                     day = x["day"],
                                     anticipacion = 1,
                                     direccion="U"), axis=1)
  """
  fecha_calculo = datetime.date(int(year), int(month), int(day)) -\
        datetime.timedelta(days=anticipacion)
  fecha_inicio = fecha_calculo - datetime.timedelta(days=1)
  fecha_fin = fecha_inicio + datetime.timedelta(days=1)
  id = id
  longitud = x
  latitud = y
  geom = ee.Geometry.Polygon([[latitud-0.1, longitud-0.1],
                              [latitud-0.1, longitud+0.1],
                              [latitud+0.1, longitud-0.1],
                              [latitud+0.1, longitud+0.1]])

  # Importar datos de google earth engine
  dataset = ee.ImageCollection('NASA/GEOS-CF/v1/rpl/htf')

  # Seleccionar el periodo de tiempo
  dataset = dataset.filter(ee.Filter.date(fecha_inicio.strftime("%Y-%m-%d"), 
                                          fecha_fin.strftime("%Y-%m-%d")))

  # Seleccionar la banda de interes
  dataset = dataset.select(direccion)

  dataset=dataset.filterBounds(geom)
  image = dataset.mean()

  fecha_inicio_str =  fecha_inicio.strftime("%Y-%m-%d")
  if direccion == "U":
    nombre_imagen = str(int(id))+"wind_north" + fecha_inicio_str
  else:
    nombre_imagen = str(int(id))+"wind_easth" + fecha_inicio_str
    
  image = image.visualize(bands=[direccion],
                          min=-30,
                          max=30,
                          palette= ["FFFFFF", "000000"])


  # Assign export parameters.
  task_config = {
      'region': geom.coordinates().getInfo(),
      'folder': 'incendios_satelite',
      'scale': 1000,
      'crs': 'EPSG:4326',
      'description': nombre_imagen
    }

  # Export Image
  task = ee.batch.Export.image.toDrive(image, **task_config)
  task.start()
