---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
(apendice)=
# Apéndice: Introducción a librerías de Python útiles


```{contents}
:local:
```

## Programación Orientada a Objetos en Python

La programación orientada a objetos (OOP, por sus siglas en inglés) es una forma de organizar código a partir de objetos. Un objeto combina datos y comportamiento: los datos suelen almacenarse en atributos, mientras que el comportamiento se implementa mediante métodos.

En Python, una clase funciona como un molde a partir del cual podemos crear objetos. Por ejemplo, si queremos representar partículas en una simulación, podemos definir una clase que describa qué información guarda cada partícula y qué operaciones puede realizar.

### Clases y objetos

Una clase se define con la palabra clave `class`. Luego, a partir de esa clase, podemos crear objetos concretos.

```{code-cell} ipython3
class Particula:
    pass


p1 = Particula()
p2 = Particula()

print(type(p1))
print(p1 is p2)
```

En este ejemplo, `Particula` es la clase, mientras que `p1` y `p2` son dos objetos distintos creados a partir de ella.

### Atributos

Los atributos son variables asociadas a un objeto. Permiten guardar información propia de cada instancia.

```{code-cell} ipython3
class Particula:
    pass


p = Particula()
p.nombre = "electrón"
p.carga = -1
p.masa = 9.11e-31

print(p.nombre)
print(p.carga)
print(p.masa)
```

Aunque Python permite crear atributos de esta manera, normalmente preferimos definirlos dentro de una función especial llamada `__init__`, que se ejecuta automáticamente al crear el objeto.

### La función `__init__`

El método `__init__` sirve para inicializar los atributos de cada objeto. Recibe como primer argumento a `self`, que representa a la instancia que se está creando.

```{code-cell} ipython3
class Particula:
    def __init__(self, nombre, carga, masa):
        self.nombre = nombre
        self.carga = carga
        self.masa = masa


electron = Particula("electrón", -1, 9.11e-31)
proton = Particula("protón", 1, 1.67e-27)

print(electron.nombre)
print(proton.masa)
```

Cada vez que escribimos `Particula(...)`, Python crea un nuevo objeto y llama automáticamente a `__init__` con los valores que pasamos.

### Métodos

Los métodos son funciones definidas dentro de una clase. Se usan para describir acciones o cálculos asociados a los objetos.

```{code-cell} ipython3
class Particula:
    def __init__(self, nombre, carga, masa):
        self.nombre = nombre
        self.carga = carga
        self.masa = masa

    def describir(self):
        print(f"{self.nombre}: carga = {self.carga}, masa = {self.masa:.2e} kg")

    def es_neutra(self):
        return self.carga == 0


neutron = Particula("neutrón", 0, 1.67e-27)
neutron.describir()
print(neutron.es_neutra())
```

Observá que `self` permite acceder a los atributos del propio objeto. Por ejemplo, dentro del método `describir`, `self.nombre` hace referencia al nombre de esa instancia en particular.

### Representación con `__repr__`

Cuando imprimimos un objeto, Python muestra por defecto una representación poco informativa. Podemos mejorar esto definiendo el método especial `__repr__`.

```{code-cell} ipython3
class Particula:
    def __init__(self, nombre, carga, masa):
        self.nombre = nombre
        self.carga = carga
        self.masa = masa


p = Particula("muón", -1, 1.88e-28)
print(p)
```

La salida indica que existe un objeto de tipo `Particula`, pero no muestra claramente su contenido. Para resolverlo:

```{code-cell} ipython3
class Particula:
    def __init__(self, nombre, carga, masa):
        self.nombre = nombre
        self.carga = carga
        self.masa = masa

    def __repr__(self):
        return f"Particula(nombre={self.nombre!r}, carga={self.carga}, masa={self.masa})"


p = Particula("muón", -1, 1.88e-28)
print(p)
```

Ahora la representación es mucho más útil, especialmente al inspeccionar objetos en una notebook o durante la depuración.

### Ejemplo completo

El siguiente ejemplo reúne los conceptos principales: clase, atributos, `__init__`, métodos y `__repr__`.

```{code-cell} ipython3
class Particula:
    def __init__(self, nombre, carga, masa):
        self.nombre = nombre
        self.carga = carga
        self.masa = masa

    def energia_en_reposo(self):
        c = 3.0e8
        return self.masa * c**2

    def __repr__(self):
        return f"Particula(nombre={self.nombre!r}, carga={self.carga}, masa={self.masa})"


electron = Particula("electrón", -1, 9.11e-31)

print(electron)
print(f"Energía en reposo: {electron.energia_en_reposo():.3e} J")
```

En muchos problemas científicos, este enfoque resulta útil porque permite representar entidades del problema, como partículas, sensores, mediciones o simulaciones, agrupando en un mismo lugar tanto sus propiedades como las operaciones que les corresponden.

## Numpy

NumPy (Numerical Python) es una biblioteca fundamental para el cálculo numérico en Python, especialmente relevante para estudiantes de física y disciplinas afines. Proporciona soporte para arrays y matrices multidimensionales, junto con una colección de funciones matemáticas de alto nivel para operar con estos datos. NumPy es esencial para realizar cálculos eficientes y es la base sobre la cual se construyen muchas otras bibliotecas científicas en Python, como SciPy, Pandas y Matplotlib.

### Arrays y Operaciones Básicas

En el corazón de NumPy se encuentra el objeto ndarray, que representa un array n-dimensional. A diferencia de las listas de Python, los arrays de NumPy son más eficientes en términos de memoria y permiten realizar operaciones matemáticas de manera vectorizada, lo que significa que las operaciones se aplican a todos los elementos del array simultáneamente, sin necesidad de bucles explícitos.

```python
import numpy as np

# Crear un array unidimensional
a = np.array([1, 2, 3, 4, 5])

# Crear un array bidimensional (matriz)
b = np.array([[1, 2, 3], [4, 5, 6]])

# Operaciones básicas
suma = a + 10  # Sumar 10 a cada elemento
producto = a * 2  # Multiplicar cada elemento por 2
```

### Funciones Matemáticas
NumPy ofrece una amplia gama de funciones matemáticas que se aplican de manera eficiente a los arrays. Esto incluye funciones trigonométricas, exponenciales, logarítmicas y estadísticas.

```python
# Funciones matemáticas
sin_a = np.sin(a)  # Seno de cada elemento
log_b = np.log(b)  # Logaritmo natural de cada elemento

# Estadísticas básicas
media = np.mean(a)  # Media de los elementos
desviacion = np.std(a)  # Desviación estándar
```

### Manipulación de Arrays
NumPy permite manipular arrays de diversas maneras, como cambiar su forma, apilarlos, dividirlos y transponerlos.

```python 
# Cambiar la forma de un array
c = np.arange(12)  # Crear un array de 0 a 11
c_reshaped = c.reshape(3, 4)  # Cambiar la forma a 3x4

# Apilar arrays
d = np.array([6, 7, 8])
apilado = np.vstack((a, d))  # Apilar verticalmente

# Transponer un array
b_transpuesta = b.T
```

### Indexación y Slicing
La indexación y el slicing en NumPy son similares a las listas de Python, pero con mayor flexibilidad para arrays multidimensionales.

```python 
# Indexación
elemento = b[1, 2]  # Elemento en la segunda fila, tercera columna

# Slicing
sub_array = b[:, 1:3]  # Todas las filas, columnas 1 a 2
```

## Matplotlib

Matplotlib es una biblioteca de Python ampliamente utilizada para la visualización de datos. Es especialmente útil para estudiantes de física que necesitan representar gráficamente resultados experimentales, simulaciones y análisis de datos. Matplotlib proporciona una amplia gama de herramientas para crear gráficos estáticos, animados e interactivos en Python.

### Características Principales

Matplotlib es conocida por su flexibilidad y capacidad para crear una variedad de tipos de gráficos, desde simples gráficos de líneas hasta complejas visualizaciones tridimensionales. Algunas de sus características clave incluyen:

- **Gráficos de Líneas y Dispersión**: Ideal para representar series temporales, funciones matemáticas y relaciones entre variables.

- **Histogramas y Gráficos de Barras**: Útiles para mostrar distribuciones de datos y comparaciones categóricas.

- **Gráficos de Sectores y Diagramas de Caja**: Permiten visualizar proporciones y distribuciones estadísticas.

- **Visualizaciones 3D**: Con el módulo `mpl_toolkits.mplot3d`, es posible crear gráficos tridimensionales para representar datos en tres dimensiones.

- **Personalización Extensiva**: Matplotlib ofrece una amplia gama de opciones para personalizar gráficos, incluyendo colores, estilos de línea, etiquetas y leyendas.

### Flujo de Trabajo Típico

El flujo de trabajo típico en Matplotlib para crear un gráfico sigue estos pasos:

1. **Importar la Biblioteca**: Importar Matplotlib y sus submódulos necesarios, como `pyplot`.

2. **Preparar los Datos**: Organizar los datos que se desean visualizar.

3. **Crear el Gráfico**: Utilizar funciones de `pyplot` para crear el tipo de gráfico deseado.

4. **Personalizar el Gráfico**: Añadir títulos, etiquetas, leyendas y ajustar el estilo del gráfico según sea necesario.

5. **Mostrar o Guardar el Gráfico**: Utilizar `show()` para visualizar el gráfico o `savefig()` para guardarlo en un archivo.

### Ejemplo Básico

A continuación, se presenta un ejemplo simple de cómo usar Matplotlib para crear un gráfico de líneas:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

# Generar datos
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Crear el gráfico
plt.figure(figsize=(8, 4))
plt.plot(x, y, label='Seno', color='b', linestyle='-')

# Personalizar el gráfico
plt.title('Gráfico de la Función Seno')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()
```

## Pandas

Pandas es una biblioteca de Python diseñada para facilitar el análisis y la manipulación de datos, y es especialmente útil para estudiantes de física que trabajan con grandes conjuntos de datos experimentales o simulados. Construida sobre NumPy, Pandas proporciona estructuras de datos y funciones de alto nivel que simplifican tareas comunes de manipulación de datos, como la limpieza, transformación y agregación.

### Estructuras de Datos Principales
Pandas introduce dos estructuras de datos fundamentales: Series y DataFrame.

-  Series: Es una estructura unidimensional similar a un array, lista o columna en una tabla. Cada elemento en una Series tiene un índice asociado, lo que permite un acceso más flexible a los datos.

```{code-cell} ipython3 
import pandas as pd

# Crear una Series
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
s
```

- DataFrame: Es una estructura bidimensional, similar a una hoja de cálculo o una tabla SQL, que contiene filas y columnas. Los DataFrames son el núcleo de Pandas y permiten almacenar y manipular datos tabulares de manera eficiente.

```{code-cell} ipython3
# Crear un DataFrame
data = {'Temperatura': [22, 21, 19, 23],
        'Humedad': [30, 45, 50, 40]}
df = pd.DataFrame(data, index=['Día 1', 'Día 2', 'Día 3', 'Día 4'])
df
```

### Manipulación de Datos

Pandas ofrece una amplia gama de funciones para manipular y transformar datos, lo que facilita tareas comunes en el análisis de datos.

- **Selección y Filtrado**: Puedes seleccionar columnas, filas o valores específicos utilizando etiquetas o condiciones.

```{code-cell} ipython3
# Seleccionar una columna
temperatura = df['Temperatura']
temperatura
```

```{code-cell} ipython3
# Filtrar filas basadas en una condición
alta_humedad = df[df['Humedad'] > 40]
alta_humedad
```

- **Operaciones de Agregación**: Pandas permite realizar operaciones de agregación como suma, media y conteo de manera sencilla.

```{code-cell} ipython3
# Calcular la media de cada columna
media_columnas = df.mean()
print(media_columnas)
```

```{code-cell} ipython3
# Sumar todos los valores de una columna
suma_temperatura = df['Temperatura'].sum()
print(suma_temperatura)
```

- **Manejo de Datos Faltantes**: Pandas proporciona métodos para identificar y manejar datos faltantes, lo cual es crucial en el análisis de datos reales.

```{code-cell} ipython3
# Identificar valores faltantes
faltantes = df.isnull()
faltantes
```

```{code-cell} ipython3
# Rellenar valores faltantes
df_rellenado = df.fillna(0)
df_rellenado
```

### Operaciones Avanzadas

- **Agrupación**: La función `groupby` permite agrupar datos y aplicar funciones de agregación a cada grupo.

```{code-cell} ipython3
# Agrupar por una columna y calcular la media
grupo = df.groupby('Humedad').mean()
grupo
```

- **Combinación de Datos**: Pandas facilita la combinación de múltiples `DataFrames` mediante operaciones de concatenación y fusión.

```{code-cell} ipython3
# Concatenar DataFrames
df_concatenado = pd.concat([df, df])
df_concatenado
```

```{code-cell} ipython3
# Fusionar DataFrames
df_otra = pd.DataFrame({'Humedad': [30, 45], 'Presión': [1012, 1015]})
df_fusionado = pd.merge(df, df_otra, on='Humedad')
df_fusionado
```

## Scikit-learn

Scikit-learn es una biblioteca de Python ampliamente utilizada para el aprendizaje automático y el análisis predictivo. Es especialmente valiosa para estudiantes de física que desean aplicar técnicas de machine learning a problemas científicos, ya que ofrece una amplia gama de herramientas para modelado predictivo, desde algoritmos de clasificación y regresión hasta técnicas de reducción de dimensionalidad y agrupamiento.

### Características Principales

Scikit-learn se destaca por su simplicidad y consistencia en la interfaz, lo que facilita su uso incluso para aquellos que son nuevos en el aprendizaje automático. A continuación, se presentan algunas de las funcionalidades clave que ofrece:

- **Modelos de Clasificación**: Scikit-learn incluye una variedad de algoritmos de clasificación, como máquinas de soporte vectorial (SVM), árboles de decisión, k-vecinos más cercanos (k-NN) y clasificadores bayesianos.

- **Modelos de Regresión**: Para problemas de predicción continua, la biblioteca ofrece algoritmos de regresión lineal, regresión logística, y regresión de soporte vectorial, entre otros.

- **Agrupamiento**: Scikit-learn proporciona técnicas de agrupamiento como k-means, clustering jerárquico y DBSCAN, útiles para identificar patrones en datos no etiquetados.

- **Reducción de Dimensionalidad**: Herramientas como el Análisis de Componentes Principales (PCA) y el Análisis Discriminante Lineal (LDA) ayudan a reducir la dimensionalidad de los datos, lo que es útil para la visualización y la reducción de ruido.

- **Preprocesamiento de Datos**: Incluye funciones para escalar, normalizar y transformar datos, lo cual es crucial para preparar los datos antes de aplicar modelos de aprendizaje automático.

### Flujo de Trabajo Típico

El flujo de trabajo típico en Scikit-learn sigue una serie de pasos bien definidos, lo que facilita la implementación de modelos de aprendizaje automático:

1. **Carga y Preparación de Datos**: Importar los datos y realizar cualquier preprocesamiento necesario, como la normalización o el manejo de valores faltantes.

2. **División de Datos**: Dividir los datos en conjuntos de entrenamiento y prueba para evaluar el rendimiento del modelo.

3. **Selección y Entrenamiento del Modelo**: Elegir un algoritmo adecuado y entrenar el modelo con los datos de entrenamiento.

4. **Evaluación del Modelo**: Evaluar el rendimiento del modelo utilizando métricas adecuadas, como precisión, recall, F1-score para clasificación, o error cuadrático medio para regresión.

5. **Ajuste de Hiperparámetros**: Utilizar técnicas como la búsqueda en cuadrícula (Grid Search) o la búsqueda aleatoria (Random Search) para optimizar los hiperparámetros del modelo.

### Ejemplo Básico

A continuación, se presenta un ejemplo simple de cómo usar Scikit-learn para un problema de clasificación:

```{code-cell} ipython3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos Iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")
```

## Pytorch

PyTorch es una biblioteca de Python de código abierto que se utiliza principalmente para el aprendizaje profundo. Desarrollada por Facebook's AI Research lab (FAIR), PyTorch es especialmente popular en la comunidad de investigación debido a su flexibilidad y facilidad de uso. Para estudiantes de física, PyTorch ofrece herramientas poderosas para construir y entrenar modelos de redes neuronales, lo que permite abordar problemas complejos de modelado y simulación.

### Características Principales

PyTorch se distingue por varias características que lo hacen atractivo para el aprendizaje profundo y la investigación científica:

- **Tensores**: Al igual que los arrays de NumPy, los tensores de PyTorch son estructuras de datos multidimensionales que permiten realizar cálculos numéricos de manera eficiente. Los tensores pueden ser manipulados en la CPU o en la GPU, lo que acelera significativamente los cálculos.

- **Autograd**: PyTorch incluye un sistema de diferenciación automática que facilita el cálculo de gradientes, esencial para el entrenamiento de redes neuronales mediante el algoritmo de retropropagación.

- **Redes Neuronales**: La biblioteca `torch.nn` proporciona módulos y funciones para construir arquitecturas de redes neuronales de manera modular y flexible.

- **Optimización**: PyTorch ofrece una variedad de algoritmos de optimización, como SGD, Adam y RMSprop, que son fundamentales para ajustar los parámetros de los modelos durante el entrenamiento.

- **Interactividad y Flexibilidad**: A diferencia de otras bibliotecas de aprendizaje profundo, PyTorch permite definir modelos de manera dinámica, lo que facilita la depuración y experimentación.

Para más detalle en cómo utilizar `PyTorch`, avanzar a la [Guia introductoria a Pytorch](./teorico_practicos/intro_pytorch.ipynb).

### Flujo de Trabajo Típico

El flujo de trabajo en PyTorch para construir y entrenar un modelo de aprendizaje profundo generalmente sigue estos pasos:

1. **Definición del Modelo**: Crear una clase que defina la arquitectura de la red neuronal utilizando los módulos de `torch.nn`.

2. **Definición de la Función de Pérdida y el Optimizador**: Elegir una función de pérdida adecuada y un algoritmo de optimización para el entrenamiento.

3. **Entrenamiento del Modelo**: Implementar un bucle de entrenamiento que itere sobre los datos, calcule la pérdida, realice la retropropagación y actualice los parámetros del modelo.

4. **Evaluación del Modelo**: Evaluar el rendimiento del modelo en un conjunto de datos de prueba para verificar su capacidad de generalización.

### Ejemplo Básico

A continuación, se presenta un ejemplo simple de cómo usar PyTorch para entrenar una red neuronal en un problema de clasificación:

```{code-cell} ipython3
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Generar un conjunto de datos sintético
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir a tensores
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Crear un DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Definir el modelo
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar el modelo
for epoch in range(10):  # 10 épocas
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluar el modelo
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f"Precisión del modelo: {accuracy:.2f}")
```



