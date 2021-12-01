# deteccion-y-clasificacion-de-ataques-en-trafico-de-red-con-dnn

Hay 4 archivos importantes

1.- pcap_2_images.py:
Se revisa el directorio y sus subdirectorios, busca archivos pcap y a cada archivo pcap genera un archivo csv con información proporcionada por snort, también se genera una carpeta con imagenes .JPEG de dimension 1 x 280 pixeles. El nombre de la imagen se divide en 2 partes separadas por un guion bajo "_". Por ejemplo: 1050_1.JPEG. Donde 1050 es la fila 1050 del archivo csv generado mientras que la segunda parte "1" hace referencia a la etiqueta de la imagen, lo cual será util para su clasificación.

2.- combinacion_pcaps.py:
Si se tiene un archivo pcap el cual es trafico limpio, este se puede conmbinar con otro archivo pcap (con una botnet en este caso), para de esta manera se genere una dataset sintética. Igual que el archivo pcap_2_images.py genera un archivo csv, una carpeta con imagenes de la misma dimensión y misma nomenclatura.

3.- create_data.py:
Este archivo tiene 2 funciones.
La primera es que busca en todo el directorio todas las imagenes se preprocesasn normalizandolas aleatorizandolas, se crean el dataset de entrenamiento y de validación para posteriormente guardar todo esto en un archivo de tipo "pickle" el cual cuando es llamado carga directamente todo a la memoria evitando que se tenga que preprocesar la información repetidas veces y solo se tenga que hacer una unica vez.
La segunda función es la función el cargar el archivo pickle en memoria y cargar directamente los datos de entrada y validación.

3.- multiples_modelos_dnn.py: Este archivo utiliza el archivo create_data.py para cargar en memoria toda la dataset creada por los otros archivos en este mismo repositorio.
Se crean varios directorios

    h5_logs: Guardan los pesos de los mejores modelos
    Tensorboard_logs: Guarda los archivos para correr tensorboard
    csv_logs: Guarda los .csv de cada modelo entrenado de cada parametro en cada epoch del modelo (excepto pesos)
    cm_graphs: Almacena las imagenes de matrices de confusión.
