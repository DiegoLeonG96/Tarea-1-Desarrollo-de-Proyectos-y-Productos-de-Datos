## 1. Instalación requerimientos
Se recomienda usar ambiente conda previamente:
- conda create tarea1_venv
- conda activate tarea1_venv
Luego instalar dependencias con:
- pip install -r requirements.txt

## 2. Generación de datos
src.data.dataset contiene una clase Dataset con un método get_preprocessed_data que carga todos los datos y los aloja en el directorio data/, tanto cruda como procesada. 
En el método change_configs se le puede ingresar un diccionario con cambios de configuración, ej: cambiar mes de entrenamiento y meses de evaluación. Si no se específican cambios se asumirá entrenamiento con 2020-01 y evaluación 2020-02, 2020-03 y 2020-05

## 3. Entrenamiento
src.modeling.train contiene la clase Training con el método fit_model, lo cual entrenará el modelo y lo guardará en models/. Esta parte asume que ya se realizó el punto 2 (va a cargar los datos de entrenamiento alojados en data/processed). De la misma forma que para la generación de datos se le pueden cambiar variables de configuración con el método change_configs

## 4. Evaluación

src.modeling.predict contiene la clase Evaluation con un método eval_model, el cual tomará el modelo alojado en models/, realizará las predicciones y calculará un dataframe con los f1-score mensuales. Esta parte asume que el paso 3  y 2 ya se ejecutó (buscará archivos de evaluación). De la misma forma que para la generación de datos se le pueden cambiar variables de configuración con el método change_configs

## 5. Visualizaciones

src.visualization.plots contiene la clase PlotsGeneration con un método plot_scores para graficar los f1-scores mensuales y el método generate_numerical_drift que grafica los histogramas de todas las features numéricas en cada mes.

## 6. Ejemplo de ejecución

El notebook alojado en notebooks/testing_pipeline.ipynb contiene una prueba con celdas breves de todos los pasos anteriores y ahonda en el estudio de la baja de rendimiento desde Abril de 2020 para un entrenamiento en Enero de 2020.

IMPORTANTE: los datos no se cargaron a este repositorio, por temas de peso de archivo. Si se ejecuta el notebook funcionará sin problemas.
