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
# De Simulaciones a inferencia

```{contents}
:local:
```


## Inferencias libre de verosimilitudes y abordaje Bayesiano

A diferencia de la estadística frecuentista (que es la que usualmente utilizamos y nos referimos cuando hablamos de estadística), en donde buscamos estimadores para parámetros *poblacionales* de distribuciones que creemos que subyacen y guían las realizaciones de mediciones de los fenómenos físicos que estudiamos, en la estadística **Bayesiana** los parámetros del modelo se tratan como variables aleatorias y se describe nuestra incertidumbre sobre ellos mediante distribuciones de probabilidad. En lugar de buscar únicamente un valor puntual para un parámetro desconocido, el enfoque Bayesiano busca construir una distribución de probabilidad sobre ese parámetro que refleje lo que sabemos antes de observar los datos y cómo esa información se actualiza luego de observarlos.

La *inferencia Bayesiana* es un método para actualizar las creencias que uno tiene acerca de un parámetro desconocido utilizando lo que se observa. La formulación central viene del teorema de Bayes, que dice que la probabilidad que cierto conjunto de parámetros $\theta$ sean los que generaron los datos dado que se observaron los datos $x$ (probabilidad a posteriori, $p(\theta|x)$ que codifica la actualización a nuestras creencias debido a que vimos los datos que vimos) se puede calcular a partir de tres cantidades: la verosimilitud (la probabilidad de observar los datos dado los parámetros $\theta$, $p(x|\theta)$), la distribución a priori de los parámetros ($p(\theta)$, que define la creencia que tenemos acerca de los parámetros antes de ver los datos observados) y la evidencia (probabilidad de los datos $p(x)$). En términos matemáticos,

$$
p(\theta | x) = \frac{p(x|\theta)p(\theta)}{p(x)}.
$$

La mayoría de los métodos de inferencia Bayesiana, en los cuáles no vamos a entrar en detalle, se basan en poder evaluar **explícitamente** la verosimilitud $p(x|\theta)$. Ejemplos de modelos son la regresión lineal Bayesiana o los procesos Gaussianos. En estos métodos se intenta aproximar la distribución posterior $p(\theta|x)$ mediante métodos como MCMC o inferencia variacional. En este curso veremos un caso específico de interés, que son las redes neuronales Bayesianas, ya que involucra la parte de aprendizaje profundo que nos interesa, pero el resto lo omitiremos para centrarnos en los demás temas.

La verosimilitud $p(x|\theta)$ es el eje central de este módulo, por lo que conviene entender claramente su significado. Esta cantidad nos está diciendo qué tan probable es que observemos los datos que observamos $x$ si los parámetros fueran $\theta$. Es la probabilidad *condicional* de observar $x$ dado $\theta$. 

En muchos casos reales, la verosimilitud resulta imposible de calcularse o extremadamente difícil. Por ejemplo, podemos contar con excelentes simuladores que pueden generar datos a partir de parámetros de entrada $x_{sim} \sim p(x|\theta)$, sin embargo no somos capaces de escribir la verosimilitud de estos simuladores de forma analítica, es decir no podemos evaluar analíticamente $p(x|\theta)$. En otras palabras, podemos simular datos, pero no podemos evaluar directamente la probabilidad de los datos observados bajo el modelo. Es aquí donde la **inferencia libre de verosimilitudes** entra en juego y cumple un rol importante. Este término refiere a un conjunto de métodos que permiten inferir sin tener que calcular explícitamente la función de verosimilitud. En vez de evaluar $p(x|\theta)$, podemos generar datos simulando el modelo y comparar las simulaciones con los datos observados.

### Cómputo Bayesiano Aproximado (ABC)

Éste es el método libre de verosimilitudes más conocido y utilizado.  En este algoritmo, muestreamos parámetros $\theta$ de nuestra creencia a priori $\theta\sim p(\theta)$. Luego, simulamos datos $x_{sim}$ para los parámetros $\theta$, $x_{sim} \sim p(x|\theta)$ y luego comparamos las simulaciones con datos reales $x_{obs}$ para ver si decidimos aceptar el parámetro evaluado o no según algún criterio. Normalmente este criterio suele ser que alguna métrica de distancia no supere cierto umbral, $d(x_{sim},x_{obs}) < \epsilon$, donde $d(\cdot)$ es la métrica y $\epsilon$ la tolerancia o valor umbral. Si se cumple esta condición entonces se acepta el valor de $\theta$ para aproximar la distribución a posteriori $p(\theta|x)$ y sino se descarta. Fijarse que en ningún momento fue necesario calcular o tener una expresión para la verosimilitud en el proceso.

```{code-cell}ipython3
import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

# datos del experimento
g_real = 9.81                  # m/s^2
sigma_t = 0.02                # error típico en segundos
alturas = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])   # metros

def tiempos_ideales(g, h):
    return torch.sqrt(2.0 * h / g)

def simular_experimento(g, alturas, sigma_t):
    t_ideal = tiempos_ideales(g, alturas)
    ruido = sigma_t * torch.randn_like(alturas)
    return t_ideal + ruido

# datos observados
t_obs = simular_experimento(g_real, alturas, sigma_t)

print("Alturas (m):", alturas)
print("Tiempos observados (s):", t_obs)

# Suponemos g ~ Uniforme(5, 15) m/s^2

def sample_prior(n):
    return 5.0 + 10.0 * torch.rand(n)

# Acá usamos directamente el vector de tiempos
# y medimos distancia euclídea.

def distancia(t_sim, t_obs):
    return torch.norm(t_sim - t_obs, p=2)

# ABC
n_propuestas = 20000
epsilon = 0.06   # tolerancia

g_propuestos = sample_prior(n_propuestas)
g_aceptados = []

for g in g_propuestos:
    t_sim = simular_experimento(g, alturas, sigma_t)
    d = distancia(t_sim, t_obs)

    if d < epsilon:
        g_aceptados.append(g.item())

g_aceptados = torch.tensor(g_aceptados)

print(f"\nCantidad de propuestas: {n_propuestas}")
print(f"Cantidad de aceptadas: {len(g_aceptados)}")
print(f"Tasa de aceptación: {len(g_aceptados)/n_propuestas:.4f}")

if len(g_aceptados) > 0:
    print(f"Media posterior ABC de g: {g_aceptados.mean().item():.3f} m/s^2")
    print(f"Desvío estándar posterior ABC: {g_aceptados.std().item():.3f} m/s^2")


plt.figure(figsize=(8, 5))
plt.hist(g_propuestos.numpy(), bins=60, density=True, alpha=0.35, label="Prior")
plt.hist(g_aceptados.numpy(), bins=30, density=True, alpha=0.75, label="Posterior ABC")
plt.axvline(g_real, linestyle="--", label="g real")
plt.xlabel("g (m/s²)")
plt.ylabel("Densidad")
plt.title("Inferencia ABC para la gravedad usando tiempos de caída libre")
plt.legend()
plt.show()

# Comparación observación vs tiempos ideales del g estimado
if len(g_aceptados) > 0:
    g_est = g_aceptados.mean()
    t_est = tiempos_ideales(g_est, alturas)

    plt.figure(figsize=(8, 5))
    plt.plot(alturas.numpy(), t_obs.numpy(), "o", label="Datos observados")
    plt.plot(alturas.numpy(), tiempos_ideales(g_real, alturas).numpy(), "--", label="Modelo ideal con g real")
    plt.plot(alturas.numpy(), t_est.numpy(), "-s", label="Modelo ideal con g estimado")
    plt.xlabel("Altura (m)")
    plt.ylabel("Tiempo de caída (s)")
    plt.title("Tiempos de caída vs altura")
    plt.legend()
    plt.show()
```

## Inferencia basada en simulaciones (SBI)

El método que vimos anteriormente es costoso computacionalmente, ya que requiere simular una gran cantidad de datos y comparar cada simulación con las observaciones. Cuando los datos son de alta dimensión o los modelos son complejos, la comparación se vuelve cada vez más difícil. 

Una alternativa que viene del lado del aprendizaje profundo es utilizar redes neuronales para *aprender* directamente relaciones probabilísticas entre parámetros y datos simulados. Este enfoque es el que se denomina *inferencia basada en simulaciones* (SBI, del inglés Simulation Based Inference).

La idea central de SBI es aprovechar la capacidad de los simuladores en generar pares de parámetros y datos $(\theta, x_{sim})\sim p(\theta) p(x_{sim}|\theta)$. Generando muchos pares mediante simulación, resulta posible entrenar los modelos probabilísticos visto anteriormente que aproximen las distribuciones relevantes para la inferencia Bayesiana.  Dependiendo de qué distribución se desea aproximar, existen diferentes estrategias dentro de SBI. Podemos aproximar la distribución a posterior (Estimación Neuronal de la Posterior, NPE), la verosimilitud (Estimación Neuronal de la Verosimilitud, NLE) y también el cociente (o razón) entre la verosimilitud y la evidencia (Estimación Neuronal de la Razón, NRE), que explicaremos a continuación.

Algnas ventajas de SBI frente a métodos como ABC son que escala mejor a datos de alta dimensión, puede reutilizar simulaciones para diferentes conjuntos de datos observados, aprovecha la capacidad de representación de los modelos probabilísticos, permite realizar inferencia en simuladores complejos, entre otras. Por esta razón es que SBI se ha vuelto una herramienta muy utilizada en áreas como la cosmología, física de partículas, biología y geología. Principalmente en áreas donde existen simuladores muy fidedignos de la realidad observada.

### Estimación de densidad neuronal e inferencia amortizada

Los métodos recientes de SBI utilizan redes neuronales (neural SBI) para aprender relaciones probabilísticas entre los datos simulados $\vec x_{sim}$ y los parámetros del modelo $\theta$. Para esto, se generan pares de simulaciones $(\theta, \vec x_{sim})$ utilizando un buen simulador del modelo. 

Si se intenta aprender la distribución posterior, es decir la distribución condicional de los parámetros dada una observación $p(\theta|\vec x_{obs})$ entonces estaremos haciendo estimación neuronal de la distribución posterior (NPE, del inglés *neural posterior estimation*). Es importante notar que estamos intentando estimar una distribución para los parámetros condicional a los datos observados. Una vez entrenada utilizando simulaciones, puede ser evaluada en *cualquier* observación para obtener una aproximación de la distribución posterior de los parámetros. A esto se le llama inferencia amortizada, ya que se entrena una vez una red neuronal de inferencia, y luego se puede utilizar en inferencia para toda observación, sin tener que reentrenar si se mide una nueva observación.

A la red neuronal que entrenamos, que predice los parámetros de una distribución de probabilidad sobre $\theta$ condicionada a los datos de entrada $\vec x$ la llamaremos red de inferencia $q_{\phi}(\theta | \vec x)$, donde $\phi$ son los parámetros entrenables de la red neuronal. Por ejemplo, supongamos que la distribución posterior es aproximadamente gaussiana, entonces la red neuronal va a aprender un mapeo no linean entre los datos de entrada $\vec x$ y los parámetros de la gaussiana (la media y varianza) para los parámetros del simulador $\theta$. Para lograr esto, $q_{\phi}$ se entrena minimizando la función de costo 

$$
\mathcal L (\phi) = \mathbb E_{(\theta, \vec x)\sim p(\theta, \vec x)\left[- \log q_{\phi}(\theta | \vec x)\right]},
$$

la cual incentiva a la red a asignar probabilidad alta a los parámetros $\theta$ del conjunto de datos de entrenamiento que generaron los datos simulados $\vec x$ correspondientes. Resulta importante notar que en el momento de entrenamiento se utilizan únicamente datos simulados. Generalmente, como modelo de red neuronal para aprender esta distribución de probabilidad, se utilizan los flujos normalizadores discutidos en el módulo anterior, los cuáles son lo suficientemente flexibles como para capturar distribuciones posteriores distintas a la Gaussiana. 


### Algunos comentarios sobre los simuladores

Necesitamos simuladores estocásticos. Esto es, simuladores que tengan algún grado de aleatoriedad en el resultado para el mismo conjunto de parámetros de entrada.



```{bibliography}
:style: unsrt
:filter: docname in docnames
```