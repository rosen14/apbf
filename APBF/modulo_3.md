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
# Embebiendo física dentro de redes neuronales

**Diapositivas teórico**: 

- [Semana_3.pdf](slides/Semana_3.pdf)

- [Semana_4.pdf](slides/Semana_4.pdf)

**Notebooks teórico-práctico**: 
- [Semana03_PINNs.ipynb](./teorico_practicos/Semana03_PINNs.ipynb)

- [Semana03_intro_deepxde.ipynb](./teorico_practicos/Semana03_intro_deepxde.ipynb)

- [Semana03_opt_bayesiana.ipynb](./teorico_practicos/Semana03_opt_bayesiana.ipynb)

- [Semana03_pinn_parametrico.ipynb](./teorico_practicos/Semana03_pinn_parametrico.ipynb)

- [Semana_04_billar_FD.ipynb](./teorico_practicos/Semana_04_billar_FD.ipynb)

- [Semana_04_pendulo_FD.ipynb](./teorico_practicos/Semana_04_pendulo_FD.ipynb)

- [Semana_04_adversarial_FD.ipynb](./teorico_practicos/Semana_04_adversarial_FD.ipynb)

**Guía de trabajo práctico**: 

- [Guia_Semana_03_PINNs.ipynb](./practicos/Guia_Semana_03_PINNs.ipynb)


```{contents}
:local:
```

En los capítulos anteriores, hemos visto una introducción a las redes neuronales profundas, en donde observamos que encuentran soluciones a partir de los datos a los problemas presentados. Hemos observado además, que no pueden aprender estríctamente más allá de lo que los datos proveen de información y que cualquier extrapolación es, al igual que con cualquier otro tipo de modelo físico o estadístico, incapaz de proveer de un resultado razonable sin ningún tipo de auxilio o consideración especial para que esto suceda. No es magia, el modelo de aprendizaje automático está aprendiendo patrones *a partir* de los datos. Por ende, si somos capaces de conseguir más datos, tendremos mejor representación de la varianza del conjunto de datos y podremos conseguir un modelo que funcione mejor en mayor cantidad de circunstancias. Es por esto que los avances en aprendizaje profundo han sido fundamentales, ya que por más que se incrementasen las bases de datos, los modelos no eran capaces de ser entrenados en tiempos razonables con las computadoras existentes. 

Ahora, si queremos resolver un problema en particular en donde tenemos información extra acerca del sistema... por qué no usarla? ¿Por qué no restringir el espacio de posibles modelos a aquellos que cumplan directamente con reglas o leyes que conocemos? Estas preguntas ya se plantearon en otras áreas como en la estadística, en donde los modelos Bayesianos incluyen información *a priori* para realizar estimaciones de probabilidad *a posteriori* luego de ver los datos. Es decir, estamos sesgando las posibilidades de los modelos intencionalmente a aquellos a los que realmente nos interesan y tienen validez en el contexto del problema que se nos presenta. Volveremos a la estadística bayesiana en los capítulos 4 y 5. Por ahora, vamos a ver como hacer para que nuestros modelos de aprendizaje automático tengan en cuenta las leyes de la física con como las entendemos. 


## Redes neuronales basadas en la física (PINNs)

Las ideas detrás de las redes neuronales basadas en la física (PINNs, del inglés *Physics Informed Neural Networks*) se comenzaron a plantear en el 2017 con los trabajos de Raiisi y Karniadakis {cite}`raissi2017physicsinformeddeeplearning,raissi2017physicsinformeddeeplearning2`. Estas redes consistuyen una manera de integrar los conocimientos físicos explícitos expresados mediante ecuaciones diferenciales que gobiernan el sistema bajo estudio. Las PINNs incorporan leyes de conservación, ecuaciones constitutivas y condiciones iniciales y de frontera directamente en el proceso de entrenamiento. 

El principio fundamental de las PINNs consiste en representar la solución de un problema físico mediante una red neuronal y definir una función de pérdida que penaliza no sólo el error respecto a los datos observadors, sino también la falta de cumplimiento de las ecucaciones que gobiernan la física del problema. Las ecuaciones diferenciales en derivadas parciales (PDE) vienen escritas en forma de derivadas parciales de la solución $u$, que puede depender de varias variables, como la posición, velocidad, tiempo, etc, que abreviaremos $u(\vec{x},t)$. Podemos decir que dada una PDE para $u(\vec{x},t)$ con evolución temporal, la podemos expresar en términos de una función $\mathcal{F}$ de sus derivadas parciales 

$$
u_{t} = \mathcal{F}(u_x, u_{xx}, \dots, u_{xx\dots x}),
$$
en donde el subíndice $_x$ indica la derivada parcial con respecto a las dimensiones espaciales (que podría incluir derivadas respecto a diferentes direcciones) y el subíndice $_t$ la variación temporal.

En el caso más sencillo de PINN, la red neuronal $q(\vec{x},t)$ aproxima la solución real del problema $u(\vec{x},t)$ a partir de datos observados. Gracias a la diferenciación automática vista en el capítulo anterior, resulta posible calcular las derivadas parciales de la red neuronal $q$ con respecto a sus variables de entrada $\vec{x},t$, es decir, resulta posible calcular el gradiente $\nabla_{\vec{x},t}{q(\vec{x},t)}$. Este gran hecho permite evaluar residuos de ecuaciones diferenciales ordinarias o parciales sin necesidad de discretizaciones como mallas o métodos de elementos finitos. Este enfoque resulta particularmente atractivo en escenarios en donde los datos son escasos, ruidosos o incompletos, pero se dispone de modelos físicos bien establecidos.

Las PINNs han demostrado utilidad en una amplia variedad de aplicaciones, incluyendo dinámica de fluídos, transferencia de calor, mecánica de sólidos, electromagnetismo, e incluse en sistemas biológicos. Éstas ofrecen una formulación unificada para problemas directos e inversos, permitiendo tanto la estimación de campos físicos como la identificación de parámetros desconocidos. 


### Función de costo física

Dada solución $u$, podemos computar el residuo $R$ como

$$
R  = u_t - \mathcal{F}(u_x, u_{xx}, \dots, u_{xx\dots x}) = 0,
$$
que naturalmente debe ser cero para la solución $u$. Si ahora planteamos la misma ecuación para una red neuronal $q$ no entrenada, resulta altamente probable que el valor no sólo no sea igual a cero, sino que sea bien distinto a cero. Si quisieramos que la red neuronal aproxime a la solución de la ecuación diferencial, entonces deberíamos exigir que el residuo $R$ para $q$ sea próximo o igual a cero. Esto se alinea muy bien con la manera de entrenar redes neuronales que hemos visto anteriormente, la función de costo. Podemos entrenar para minimizar este residuo en combinación con los términos de costo de aprendizaje automático tradicional, como MSE, MAE, etc. anteriormente vistos. Más aún, a medida que vamos aproximando la solución $u$ mediante $q$, podemos evaluar a la aproximación en puntos específicos ($\vec{x_0},\vec{x_n}$) en donde queremos que la solución cumpla con determinadas condiciones (como condiciones de contorno, o condiciones iniciales) y podemos comparar con la solución $u(\vec{x_i},t_i)=y_i$ para generar más términos de costo de la forma $q(\vec{x_i},t_i)-y_i$ que queremos minimizar. De esta forma nuestra función de costo objetivo de entrenamiento se puede escribir como

$$
\text{arg min}_{\theta} \sum_i \alpha_0 (q(\vec{x_i},t_i)-y_i)^2 + \alpha_1 R(x_i)
$$
en donde $\alpha_{0,1}$ denotan hiperparámetros que escalean la contribución del término supervisado y del residual respectivamente. Estos serían los términos de costo física descriptos, pero podría haber terminos de costo adicionales con sus factores de escala correspondientes.

Entendamos la ecuación anterior. El primer término es un término convencional, una función de costo L2. Si optimizacemos éste término solamente, la red neuronal aproximaría a las muestras de entrenamiento correctamente, pero podría promediar múltiples modos en las soluciones, funcionando erróneamente en regiones entre los puntos muestreados. Si, en cambio, optimizamos sólamente el segundo término (el residual físico), puede que la red neuronal satisfaga localmente la PDE pero que tenga dificultades encontrando una solución que ajuste globalmente. Esto puede suceder debido a que pueden haber muchas soluciones que satisfagan el mismo residual. Cuando se optimizan ambos términos en simultáneo, la red aprende a aproximar una solución específica a los datos de entrenamiento mientras que captura el conocimiento subyacente en la PDE. 

Notar que no tenemos ninguna garantía que el término residual alcance cero durante el entrenamiento. Es decir que este tratamiento propuesto impone restricciones suaves (o *soft constraints*), sin ninguna garantía del cumplimiento del constraint, sino más bien de aproximarse hacia ello. 

Planteando el problema de esta manera, estamos pensando en una representación neuronal de campo que llamaremos campo neuronal. Es decir, nuestra red neuronal $q$ está siendo optimizada de manera tal de satifacer $R=0$. Por lo tanto $q=q(\vec{x},\theta)$, donde elegimos los parámetros $\theta$ de la red de manera tal que $q\approx u$ lo más posible. A esto se le suele llamar red neuronal basada en la física (PINN, del inglés physics informed nueral network) y el artículo {cite}`raissi2019physics` sirve de excelente guía para entender cómo se utilizan en problemas inversos y hacia adelante.

### Ejemplo: Optimización de la ecuación de Burger con una PINN

Consideremos la tarea de reconstrucción como un problema inverso. Vamos a utilizar la ecuación de Burger 

$$
\frac{\partial u}{\partial t} + u \nabla u = \nu \nabla \cdot \nabla u,
$$
una ecuación simple pero a la vez no lineal en 1 dimensión. Supongamos que contamos con una serie de observaciones a tiempo $t=0.5$. La solución debe cumplir con el residual de la formulación de la ecuación de Burger como a su vez coincidir con las observaciones en los puntos medidos. A su vez, imponemos la condición de contorno de Dirichlet $u=0$ en los bordes del dominio computacional y definimos la solución en el intervalo de tiempo $t\in [0,1]$.


```{code-cell} ipython3
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Network
# ----------------------------
class Network(nn.Module):
    def __init__(self, hidden=20, depth=8):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        if x.shape != t.shape:
            raise ValueError(f"x and t must have same shape, got {x.shape} vs {t.shape}")
        y = torch.stack([x, t], dim=-1)      # (..., 2)
        y_flat = y.reshape(-1, 2)            # (M, 2)
        out_flat = self.net(y_flat)          # (M, 1)
        out = out_flat.reshape(*x.shape, 1) # (..., 1)
        return out


_model = Network().to(device)

def network(x, t):
    return _model(x, t)

# ----------------------------
# Boundary / sampling utilities
# ----------------------------
def boundary_tx(N, device=None, dtype=torch.float32):
    x = torch.linspace(-1, 1, 128, device=device, dtype=dtype)
    u = torch.tensor( [0.008612174447657694, 0.02584669669548606, 0.043136357266407785, 0.060491074685516746, 0.07793926183951633, 0.0954779141740818, 0.11311894389663882, 0.1308497114054023, 0.14867023658641343, 0.1665634396808965, 0.18452263429574314, 0.20253084411376132, 0.22057828799835133, 0.23865132431365316, 0.25673879161339097, 0.27483167307082423, 0.2929182325574904, 0.3109944766354339, 0.3290477753208284, 0.34707880794585116, 0.36507311960102307, 0.38303584302507954, 0.40094962955534186, 0.4188235294008765, 0.4366357052408043, 0.45439856841363885, 0.4720845505219581, 0.4897081943759776, 0.5072391070000235, 0.5247011051514834, 0.542067187709797, 0.5593576751669057, 0.5765465453632126, 0.5936507311857876, 0.6106452944663003, 0.6275435911624945, 0.6443221318186165, 0.6609900633731869, 0.67752574922899, 0.6939334022562877, 0.7101938106059631, 0.7263049537163667, 0.7422506131457406, 0.7580207366534812, 0.7736033721649875, 0.7889776974379873, 0.8041371279965555, 0.8190465276590387, 0.8337064887158392, 0.8480617965162781, 0.8621229412131242, 0.8758057344502199, 0.8891341984763013, 0.9019806505391214, 0.9143881632159129, 0.9261597966464793, 0.9373647624856912, 0.9476871303793314, 0.9572273019669029, 0.9654367940878237, 0.9724097482283165, 0.9767381835635638, 0.9669484658390122, 0.659083299684951, -0.659083180712816, -0.9669485121167052, -0.9767382069792288, -0.9724097635533602, -0.9654367970450167, -0.9572273263645859, -0.9476871280825523, -0.9373647681120841, -0.9261598056102645, -0.9143881718456056, -0.9019807055316369, -0.8891341634240081, -0.8758057205293912, -0.8621229450911845, -0.8480618138204272, -0.833706571569058, -0.8190466131476127, -0.8041372124868691, -0.7889777195422356, -0.7736033858767385, -0.758020740007683, -0.7422507481169578, -0.7263049162371344, -0.7101938950789042, -0.6939334061553678, -0.677525822052029, -0.6609901538934517, -0.6443222327338847, -0.6275436932970322, -0.6106454472814152, -0.5936507836778451, -0.5765466491708988, -0.5593578078967361, -0.5420672759411125, -0.5247011730988912, -0.5072391580614087, -0.4897082914472909, -0.47208460952428394, -0.4543985995006753, -0.4366355580500639, -0.41882350871539187, -0.40094955631843376, -0.38303594105786365, -0.36507302109186685, -0.3470786936847069, -0.3290476440540586, -0.31099441589505206, -0.2929180880304103, -0.27483158663081614, -0.2567388003912687, -0.2386513127155433, -0.22057831776499126, -0.20253089403524566, -0.18452269630486776, -0.1665634500729787, -0.14867027528284874, -0.13084990929476334, -0.1131191325854089, -0.09547794429803691, -0.07793928430794522, -0.06049114408297565, -0.0431364527809777, -0.025846763281087953, -0.00861212501518312], device=device, dtype=dtype)
    t = torch.full_like(x, 0.5)
    perm = torch.randperm(128, device=device)
    return x[perm][:N], t[perm][:N], u[perm][:N]

def open_boundary(N, device=None, dtype=torch.float32):
    t = torch.rand(N, device=device, dtype=dtype)
    half = N // 2
    x = torch.cat(
        [torch.ones(half, device=device, dtype=dtype),
         -torch.ones(half, device=device, dtype=dtype)],
        dim=0
    )
    u = torch.zeros(N, device=device, dtype=dtype)
    return x, t, u

# ----------------------------
# Autograd helpers
# ----------------------------
def gradients(y, x):
    return torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]

def burgers_residual(u, x, t):
    u_t  = gradients(u, t)
    u_x  = gradients(u, x)
    u_xx = gradients(u_x, x)
    return u_t + u * u_x - (0.01 / np.pi) * u_xx

# ----------------------------
# Grid for visualization
# ----------------------------
N = 128
grids_xt = np.meshgrid(
    np.linspace(-1, 1, N),
    np.linspace(0, 1, 33),
    indexing="ij"
)
grid_x = torch.tensor(grids_xt[0], dtype=torch.float32, device=device)
grid_t = torch.tensor(grids_xt[1], dtype=torch.float32, device=device)

with torch.no_grad():
    grid_u = network(grid_x, grid_t).unsqueeze(0)

# ----------------------------
# Visualization helper
# ----------------------------
# ----------------------------
# Visualization helper
# ----------------------------
def show_state(a, title):
    U = a[0, :, :, 0].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(16, 5))
    im = plt.imshow(
    U,
    origin="upper",
    aspect="auto",
    cmap="inferno",
    extent=[0, 1, -1, 1]
    )
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("time")
    ax.set_ylabel("x")
    ax.set_title(title)
    plt.show()

print("Randomly initialized network state:")
show_state(grid_u, "Uninitialized NN")

# ----------------------------
# Optimizer
# ----------------------------
optimizer = torch.optim.Adam(_model.parameters(), lr=1e-3)

# ----------------------------
# Boundary data
# ----------------------------
N_SAMPLE_POINTS_BND = 100
x_t0, t_t0, u_t0 = boundary_tx(N_SAMPLE_POINTS_BND, device=device)
x_ob, t_ob, u_ob = open_boundary(N_SAMPLE_POINTS_BND, device=device)

x_bc = torch.cat([x_t0, x_ob], dim=0)
t_bc = torch.cat([t_t0, t_ob], dim=0)
u_bc = torch.cat([u_t0, u_ob], dim=0)

# ----------------------------
# Training loop
# ----------------------------
N_SAMPLE_POINTS_INNER = 1000
ITERS = 10_000
ph_factor = 1.0

start = time.time()

for step in range(ITERS + 1):
    optimizer.zero_grad()

    # Boundary loss
    u_pred_bc = network(x_bc, t_bc).reshape(-1)
    loss_u = torch.mean((u_pred_bc - u_bc.reshape(-1)) ** 2)

    # Physics loss (resample each iter)
    x_ph = (2.0 * torch.rand(N_SAMPLE_POINTS_INNER, device=device) - 1.0).requires_grad_(True)
    t_ph = torch.rand(N_SAMPLE_POINTS_INNER, device=device).requires_grad_(True)

    u_pred_ph = network(x_ph, t_ph).reshape(-1)
    residual = burgers_residual(u_pred_ph, x_ph, t_ph)
    loss_ph = torch.mean(residual ** 2)

    loss = loss_u + ph_factor * loss_ph
    loss.backward()
    optimizer.step()

    if step < 3 or step % 1000 == 0:
        with torch.no_grad():
            grad_norm = torch.sqrt(
                sum((p.grad**2).sum() for p in _model.parameters() if p.grad is not None)
            )
        print(
            f"Step {step:6d} | "
            f"Total: {loss.item():.6e} | "
            f"Boundary: {loss_u.item():.6e} | "
            f"Physics: {loss_ph.item():.6e} | "
            f"|grad|: {grad_norm.item():.3e}"
        )
end = time.time()
print(f"Runtime {end - start:.2f}s")

# ----------------------------
# After training visualization
# ----------------------------
_model.eval()
with torch.no_grad():
    grid_u = network(grid_x, grid_t).unsqueeze(0)

show_state(grid_u, "After Training")

```
<!-- ## Diferenciación automática para ecuaciones diferenciales parciales con constraints

Las redes neuronales soportan inherentemente el cálculo de las derivadas con respecto al vector de entrada. La derivada $\partial f / \partial \theta$ es un ingrediente fundamental para aprender por el descenso por el gradiente. 

Si el vector de entrada es $\vec{x}$ -->

## Física diferenciable

En esta sección vamos a explorar la posibilidad de incorporar simulaciones numéricas diferenciables al proceso de aprendizaje, lo cuál abreviaremos como física diferenciable (FD). 

El objetivo central de esta metodología es aprovechar los algorítmos numéricos existentes para resolver ecuaciones diferenciales para mejorar los sistemas de inteligencia artificial. Para esto, resulta necesario equipar a los sistemas de IA con la funcionalidad de calcular los gradientes de un simulador físico con respecto a sus entradas, es decir, que puedan calcular cómo cambian las salidas del simulador cuando cambian sus entradas o parámetros, a modo de optimizar o aprender esos parámetros usando gradientes. Una vez que hacemos esto para todos los operadores de una simulación, podemos utilizar la diferenciación automática {cite}`baydin2018automatic` a nuestro favor junto con la retropropagación para permitir que la información del gradiente fluya del simulador a la red neuronal y viceversa. Lo que estamos haciendo es integrar directamente el simulador físico dentro del grafo computacional del modelo de aprendizaje.

En contraste con las PINNs propuestas anteriormente, esto permite manejar espacios de soluciones más complejos, sin necesidad de aprender para un problema inverso específico como hicimos anteriormente. La física diferenciable permite entrenar redes neuronales que aprender a aproximar soluciones a un conjunto mayor de problemas inversos de manera más eficiente. Se trata de usar décadas de conocimiento en métodos numericos y hacerlos compatibles con aprendizaje por gradiente. Es decir, el solver sigue tiendo el experto físico pero la red neuronal aprende alrededor o encima de él.

Para clarificar, un simulador físico clásico agarra una entrada y provee de una salida que resulta de la descripción física del simulador. Por ejemplo la entrada puede ser alguna cantidad vectorial inicial y algunos parámetros que escalean cierta ecuación, $(\vec{u}(t=0), \nu)$ y la salida puede ser el vector evolucionado a tiempo $T$, $\vec{u}(t=T)$. Sin embargo, estos solvers no nos dicen cómo cambiar la entrada para mejorar el resultado (donde mejorar el resultado resta por definir y será definido bajo cierto objetivo). En FD, el simulador sabe cómo calcular $\partial \vec u (T) / \partial \vec{u}(t=0)$ y $\partial \vec u (T) / \partial \nu$, lo que permite insertar al simulador dentro del pipeline de aprendizaje, propagar gradientes a través del simulador, y por lo tanto optimizar parámetros, estados iniciales o controles. 

Al poder calcular los gradientes, el simulador se convierte en una "capa más del modelo", con lo que los frameworks de aprendizaje profundo pueden usar retropropagación y permitir que los gradientes fluyan desde la función de costo a través del simulador y hacia una red neuronal o hacia parámetros físicos, permitiendo entrenar sistemas híbridos. La interacción de los sitemas híbridos puede ser de la red hacia el simulador o viceversa. En el primer caso, la red puede generar parámetros físicos, condiciones inciales, fuerzas externas, etc. y el simulador puede usar esto para producir una evolución física. En el segundo caso, la red recibe estados simulados, gradientes físicos e información estructurada del simulador y puede a partir de eso generar una predicción. 

A continuación veremos cómo se vuelven diferenciales los solvers existentes.

### Operadores diferenciables

En FD trabajamos encima de solvers numéricos existentes. Depende fuertemente en los algoritmos computacionales ya disponibles en cada área de la física. Para comenzar, necesitamos una formulación contínua como modelo para el efecto física que queremos simular. 

Asumimos que tenemos una formulación contínua $\mathcal{P}^*(\vec{u}, \nu)$ de la cantidad física de interés $\vec{u}(\vec{u},t) : \mathbb{R}^d\times \mathbb{R}^+ \rightarrow \mathbb{R}^d$, con parámetros de modelo $\nu$ (por ejemplo, constante de difusión, viscocidad, conductividad, etc.). Las componentes de $u$ estarán denotadas por un  un sub-índice $i$ ($\vec{u}=(u_1,\dots ,u_d)^T$). Típicamente estamos interesados en la evolución temporal de $\vec{u}$. Discretizamos en intervalos de tiempo $\Delta t$ y esto resulta en una formulación $\mathcal{P}(\vec{u},\nu)$. El estado a tiempo $t+\Delta t$ se computa mediante la secuencia de operadores $\vec{\mathcal{P}_1, \mathcal{P}_2, \dots, \mathcal{P}_m}$ de tal manera que $\vec{u}(t+\Delta t) = \mathcal{P}_m \circ \dots \circ \mathcal{P}_2 \circ \mathcal{P}_1(\vec{u}(t), \nu)$, donde $\circ$ denota la composición de funcioines, i.e. $f\circ g(x) = f(g(x))$. 

Para incorporar este solver numérico al proces de aprendizaje profundo, necesitamos contar con el gradiente de cada operador $\mathcal{P}_i$ con respecto a sus inputs, i.e. $\partial \mathcal{P}_i/\partial \vec{u}$. Notar que no necesitamos siempre las derivadas con respecto a todos los parámetros (por ejemplo tal vez no queremos optimizar con respecto de $\nu$), con lo cual se pueden omitir ciertas derivadas. En lo que sigue asumimos que $\nu$ es un parámetro del modelo, pero que no va a ser una de las salidas de nuestra red neuronal, para evitar pasar $\partial \mathcal{P}_i/\partial \vec{\nu}$ a nuestro solver numérico.

### Jacobianos

Como típicamente $\vec{u}$ es un vector, $\partial \mathcal{P}_i/\partial \vec{u}$ denota una matriz Jacobiana $J$ en vez de un único valor, i.e. 

$$
\frac{\partial \mathcal{P}_i}{\partial \vec{u}} = \nabla_{\vec{u}}\mathcal{P}_i = \begin{bmatrix} 
    \partial \mathcal P_{i,1} / \partial u_{1} 
    & \  \cdots \ &
    \partial \mathcal P_{i,1} / \partial u_{d} 
    \\
    \vdots & \ & \ 
    \\
    \partial \mathcal P_{i,d} / \partial u_{1} 
    & \  \cdots \ &
    \partial \mathcal P_{i,d} / \partial u_{d} 
    \end{bmatrix},
$$
en donde $d$ denota el número de componentes de $\vec{u}$. En este caso, como $\mathcal{P}$ mapea un valor de $\vec{u}$ a otro valor de de $\vec{u}$, el jacobiano es cuadrado, pero podría ser que este no sea el caso sin traer ningún tipo de problema a la metodología propuesta. 

Ahora utilizaremos el modo reverso de diferenciación automática y nos centraremos en computar un producto vectorial de matrices entre la transpuesta del jacobiano y un vector $\vec{a}$, .i.e. $\left(\frac{\partial \mathcal{P}_i}{\partial \vec{u}}\right)^T \vec{a}$. Si tuviesemos que construir y almacenar toda matriz jacobiana que necesitamos durante el entrenamiento causaría mucho uso de memoria y relentizaría el proceso de entrenamiento innecesariamente. En vez de eso, en la retropropagación, podemos computar productos con el jacobiano más rápidos porque siempre tenemos una función escalar de costo al final de la cadena. 


Teniendo en cuenta esta formulación, necesitamos resolver la derivada de la función de costo escalar $\mathcal l$, lo cual es equivalente a considerar las derivadas de la cadena de funciones compuestas $\mathcal{P}_i$ evaluadas en un estado actual dado $\vec{u}^n$ mediante la regla de la cadena. A modo de ejemplo, para el caso de dos operadores


$$
    \frac{\partial \mathcal l}{ \partial \mathbf{u}}  = \frac{ \partial (\mathcal P_2 \circ \mathcal P_1) }{ \partial \mathbf{u} } \Big|_{\mathbf{u}^n}
    = 
    \frac{ \partial \mathcal P_2 }{ \partial \mathcal P_1 } \big|_{\mathcal P_1(\mathbf{u}^n)}
    \ 
    \frac{ \partial \mathcal P_1 }{ \partial \mathbf{u} } \big|_{\mathbf{u}^n} \ ,
$$
o cual corresponde a la versión vectorial de la regla de la cadena clásica y se extiende de forma directa al caso de una composición de más de dos operadores $i>2$.

Las derivadas de $\mathcal{P}_1$ y $\mathcal{P}_2$ siguen siendo jacobianos, pero dado que la función de costo $\mathcal l$ es escalar, el gradiente de $\mathcal l$ con respecto al último operador de la cadena es un vector. En el modo reverso de diferenciación, se inicia la propagación con este gradiente y se calculan sucesivamente los productos Jacobiano propagando la información de sensibilidad hacia el estado inicial $\vec{u}$.


De esta manera, una vez que podemos calcular los productos de los Jacobianos de los operadores de nuestro simulador, podemos integrarlos dentro del pipeline de aprendizaje profundo, al igual que uno incluiría una capa completamente conectada o una función de activación de ReLU. Es decir, resulta totalmente compatible incluir la capa de física diferencial dentro de los modelos de aprendizaje automático. 

Uno podría pensar que, dado que la mayoría de los solvers pueden descomponerse en una secuencia de operaciones vectoriales y matriciales y dado que los frameworks de deep learning como PyTorch ya soportan este tipo de operaciones y diferenciación automática, se podrían implementar directamente los solver físicos usando estas operaciones básicas únicamente. Aunque teóricamente esto es posible, en la práctica resulta inconveniente. La principal inconveniencia es que cada operación elemental (suma, multiplicación, producto, etc.) se evalúa de manera independiente y por lo tanto debe almacenar información intermedia durante la evaluación hacia adelante para poder calcular los gradientes durante el proceso de retropropagación. En una simulación física, normalmente no estamos interesados en cada uno de estos resultados intermedios, sino sólamente cómo se actualiza el estado del sistema, pasando de $\vec{u}(t)$ hasta $\vec{u}(t+\Delta t)$. Por ende, en la práctica resulta mucho más eficiente agrupar el proceso de resolución en una secuencia de operadores grandes y significativos a los que llamaremos operadores *monolíticos*. Cada uno de estos operadores encapsula un paso relevante del método numérico completo en lugar de exponer todas las operaciones elementales que lo componen. Este enfoque tiene varias ventajas. Por un lado, evita el cálculo y almacenamiento innecesario, reduciendo el consumo de memoria y costo computacional. Por otro, permite elegir los métodos numéricos más edcuados para calcular tanto la actualización del estado como sus derivadas. Además, permite aprovechar algoritmos numéricamente eficientes (por ejemplo para inversión de matrices, se pueden usar solvers multigrid con complejidad $\mathcal O (n)$). El principal inconveniente de este enfoque es que requiere un mayor entendimiento del problema físico y de los métodos numéricos utilizados. Además, muchos solvers no propocionan de forma directa las derivadas necesarias para el aprendizaje, por lo que deben implementarse explícitamente. 

Como comentario final, en la práctica conviene ser avaro con las derivadas que se implementan. Sólo es necesario proporcionar aquellas que realmente intervienen en el proceso de aprendizaje. Es decir, si por ejemplo una red neuronal nunca genera el parámetro $\nu$ y este no aparece tampoco en la función de costo, entonces durante la retropropagación nunca será necesario calcular derivadas con respecto al parámetro (y esto ahora mucho tiempo).

### Ejemplo: Oscilador armónico forzado

El oscilador armónico amortiguado es uno de los modelos mínimos para describir sistemas dinámicos reales. En su forma más estándar, el sistema se expresa como una ecuación diferencial ordinaria de segundo orden

$$
m,\ddot{x}(t) + c,\dot{x}(t) + k,x(t) = 0,
$$

donde $x(t)$ es el desplazamiento, $m>0$ es la masa (asumida conocida en este ejemplo), $c>0$ es el coeficiente de amortiguamiento (pérdidas por fricción viscosa) y $k>0$ es la constante elástica (rigidez del resorte). 

Si $c$ y $k$ fuesen conocidos, el problema directo consiste en predecir la trayectoria $x(t)$ dada una condición inicial $(x(0),\dot{x}(0))$. Sin embargo, en aplicaciones reales ocurre el problema inverso: observamos el movimiento (tipicamente con ruido) y queremos inferir los parámetros que lo explican. A esto se le suele llamar **problema inverso**. 

Cuando medimos, casi nunca disponemos de trayectorias continuas. Observamos muestras discretas $x(t_0), x(t_1), \dots$ y terminamos usando un integrador numérico para aproximar la dinámica. Usaremos un esquema explícito simple tipo Euler {cite}`wiki:Euler_method` (actualizando velocidad y posición con un paso $dt$) para generar trayectorias discretas. En este caso, se calcula la aceleración 

$$
a = - \frac{c}{m}v - \frac{k}{m}x
$$

y luego se actualizan las velocidades y la posición de la forma $v \leftarrow v + dt\,a$ y $x \leftarrow x + dt\,v$. A continuación implementamos un simulador en `PyTorch` lo que lo hace diferenciable. Es decir, podemos calcular las derivadas vía diferenciación automática para analizar como cambian las trayectorias con respecto a los parámetros $c$ y $k$. 

```{code-cell} ipython3
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.manual_seed(0)


def simulador_oscilador_amortiguado_batch(m, c, k, x0, v0, dt, T):
    """
    c, k: tensores (B,)
    Salida x_traj: (B, T+1)
    """
    B = c.shape[0]

    x = x0.expand(B)      # (B,)
    v = v0.expand(B)      # (B,)

    xs = [x]

    for _ in range(T):
        a = -(c / m) * v - (k / m) * x   # (B,)
        v = v + dt * a
        x = x + dt * v
        xs.append(x)

    return torch.stack(xs, dim=1)  # (B, T+1)

```

Podemos visualizar el resultado del simulador de la siguiente manera.

```{code-cell} ipython3
# si entrenar en cpu o gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Constantes de simulación
m_true = torch.tensor(1.0, dtype=torch.float32, device=device) # masa conocida
x0 = torch.tensor(1.0, dtype=torch.float32, device=device) # condición inicial posición
v0 = torch.tensor(0.0, dtype=torch.float32, device=device) # condición inicial velocidad
dt = torch.tensor(0.01, dtype=torch.float32, device=device) # paso de tiempo
T = 400 # número de pasos de tiempo


xs = simulador_oscilador_amortiguado_batch(
    m=m_true,
    c=torch.tensor([0.5], dtype=torch.float32, device=device),
    k=torch.tensor([4.0], dtype=torch.float32, device=device),
    x0=x0,
    v0=v0,
    dt=dt,
    T=T
)

plt.figure(figsize=(8,4))
plt.plot(xs[0].cpu().numpy(), label='x(t)')
plt.title('Trayectoria del oscilador amortiguado')
plt.xlabel('Tiempo (pasos)')
plt.ylabel('Posición x')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
```

Ahora plantearemos una red neuronal que nos permita inferir los parámetros físicos $(c, k)$ que caracterizan al sistema dadas $M$ observaciones iniciales del desplazamiento $\{x(t_0),\dots, x(t_{M-1})\}$. La arquitectura propuesta, `RedCK`, es una red completamente conectada (MLP) que recibe un vector de longitud $M$ y produce dos salidas: una estimación de $c$ y otra de $k$. La red devuelve lo que llamé valores "raw" (o crudos) que luedo pueden transformarse para cumplir con los constraints físicos, como la positividad de ambos valores, como explicado en el capítulo {ref}`contraints_fisicos`.


```{code-cell} ipython3
class RedCK(nn.Module):
    '''
    Red neuronal para predecir c y k dados M puntos de la trayectoria.
    '''
    def __init__(self, M):
        super().__init__()
        self.fc1 = nn.Linear(M, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 2)

    def forward(self, x_partial):
        h = F.relu(self.fc1(x_partial))
        h = F.relu(self.fc2(h))
        raw = self.out(h)
        return raw[:, 0], raw[:, 1]

def positive(p):
    '''
    Asegura que p > 0 usando softplus.
    '''
    return F.softplus(p) + 1e-6
```

Para entrenar y evaluar el sistema, construiremos un conjunto de datos sintéticos que reproduce datos medidos. Para esto, muestrearemos parámetros $c$ y $k$ de distribuciones uniformes en rangos relativamente razonables. Luego simularemos la trayectoria completa $x(t)$ durante $T$ pasos de tiempo para cada set de parámetros muestreados. Agregaremos además ruido gaussiano para que asimilen más mediciones reales y dividiremos los datos generados en una parte de longitud $M$ que será el input de nuestra red, y el resto que quedará disponible como referencia.

```{code-cell} ipython3
def sample_batch(batch_size, M, T, dt, m, x0, v0, ruido_std=0.01):
    '''
    Genera un batch de datos sintéticos.

    Parámetros:
    - batch_size: número de muestras en el batch
    - M: número de observaciones iniciales con ruido
    - T: número total de pasos de tiempo
    - dt: paso de tiempo
    - m: masa del oscilador
    - x0: condición inicial posición
    - v0: condición inicial velocidad
    - ruido_std: desviación estándar del ruido gaussiano añadido

    Devuelve:
    - x_partial: (B, M) primeras M posiciones con ruido
    - y_obs: (B, T+1) posiciones completas con ruido
    - c_true: (B,) coeficientes de amortiguamiento verdaderos
    - k_true: (B,) constantes elásticas verdaderas
    '''
    c_true = torch.empty(batch_size, device=device).uniform_(0.05, 1.0)
    k_true = torch.empty(batch_size, device=device).uniform_(0.5, 8.0)

    x_traj = simulador_oscilador_amortiguado_batch(
        m, c_true, k_true, x0, v0, dt, T
    )

    y_obs = x_traj + ruido_std * torch.randn_like(x_traj)
    x_partial = y_obs[:, :M]

    return x_partial, y_obs, c_true, k_true
```

Podemos observar algunos datos para darnos una idea de la estructura de nuestros datos

```{code-cell} ipython3
def graficar_batch(batch_size=5, M=150, T=400, dt_val=0.01, m=None, x0=None, v0=None, ruido_std=0.01):
    """
    Grafica un batch de trayectorias del oscilador amortiguado.
    
    Parámetros:
    - batch_size: número de trayectorias a graficar (default 5)
    - M: número de observaciones iniciales
    - T: número total de pasos de tiempo
    - dt_val: paso de tiempo
    - m: masa del oscilador
    - x0: condición inicial posición
    - v0: condición inicial velocidad
    - ruido_std: desviación estándar del ruido
    """
    
    # Generar batch
    x_partial, y_obs, c_true, k_true = sample_batch(
        batch_size, M, T, dt_val, m, x0, v0, ruido_std=ruido_std
    )
    
    fig, axes = plt.subplots(batch_size, 1, figsize=(12, 3*batch_size))
    if batch_size == 1:
        axes = [axes]
    
    t = (torch.arange(T + 1) * dt_val).cpu()
    
    for i in range(batch_size):
        ax = axes[i]
        
        # Observaciones con ruido (primeras M)
        ax.plot(t[:M].cpu(), y_obs[i, :M].cpu(), "k.", markersize=3, alpha=0.6, label="Obs. (primeras M)")
        
        # Trayectoria completa observada
        ax.plot(t.cpu(), y_obs[i].cpu(), "b-", linewidth=1.5, alpha=0.7, label="Trayectoria obs.")
        
        # Trayectoria con parámetros verdaderos
        ax.plot(t[:M].cpu(), x_partial[i].cpu(), "r--", linewidth=2, label="Trayectoria parcial")
        
        # Línea vertical en M
        ax.axvline(t[M].item(), color="gray", linestyle=":", alpha=0.5)
        
        # Etiqueta con parámetros
        c_val = c_true[i].item()
        k_val = k_true[i].item()
        ax.set_title(f"Muestra {i+1} | c_true={c_val:.4f}, k_true={k_val:.4f}", fontsize=10)
        ax.set_ylabel("Posición x(t)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)
    
    axes[-1].set_xlabel("Tiempo")
    plt.tight_layout()
    plt.show()



# --------
# Entrenamiento
# --------
M = 150 # número de observaciones iniciales usadas como input de la red

graficar_batch(batch_size=5, M=M, T=T, dt_val=dt.item(), m=m_true, x0=x0, v0=v0)
```


Podemos definir la red, un optimizador para el proceso de optimización y la cantidad de iteraciones, así como el tamaño del batch. Esto dependerá fuertemente de los recursos computacionales con los que contemos. 


```{code-cell} ipython3
net = RedCK(M) # inicializar red
net.to(device) # mover a dispositivo

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3) # optimizador

steps = 1000 # número de iteraciones de entrenamiento
batch_size = 64 # tamaño de batch
```

Vamos a armar el loop de entrenamiento, en donde plantearemos una función de costo que presenta dos términos. El primer término tendrá en cuenta el error en la trayectoria, es decir la consistencia con la dinámica de las ecuaciones. Este término lo llamaremos $\mathcal L_{traj}$ y estará dado por

$$
\mathcal L_{traj} = \frac{1}{n_{b}n_{T}}\sum_{b,t}(\hat{x}_{b,t}-x^{obs}_{b, t})^2,
$$

en donde el subíndice $b=1,\dots,n_b$ refiere a las trayectorias del batch y $t=1,\dots,n_{T}$ a los instantes temporales.

El segundo término será el error en los parámetros, es decir el aprendizaje supervisado directo. Este término lo denotaremos $\mathcal L_{param}$ y estará dado por la suma de los errores cuadrático medio de los parámetros $c$ y $k$. Plantearemos un factor que controla el compromiso entre ambas funciones de costo para la función de costo total

$$
\mathcal L_{tot} = \mathcal{L}_{traj} + 0.5 \mathcal L_{param}.
$$

En cada iteración, la red neuronal infiere parámetros físicos a partir de las observaciones parciales y luego se evalúa el simulador diferenciable con los parámetros. El aprendizaje se producirá al minimizar la discrepancia entre la dinámica observada y la dinámica reconstruída. 


```{code-cell} ipython3
for it in range(steps):
    optimizer.zero_grad()

    x_partial, y_obs, c_true, k_true = sample_batch(
        batch_size, M, T, dt, m_true, x0, v0
    ) # generar batch sintético

    # mover a dispositivo
    x_partial = x_partial.to(device)
    y_obs = y_obs.to(device)

    raw_c, raw_k = net(x_partial) # predecir parámetros
    # asegurar positividad
    c_hat = positive(raw_c) 
    k_hat = positive(raw_k)

    x_hat = simulador_oscilador_amortiguado_batch(
        m_true, c_hat, k_hat, x0, v0, dt, T
    ) # simular con parámetros predichos

    loss_traj = torch.mean((x_hat - y_obs) ** 2) # costo MSE de trayectoria
    loss_param = torch.mean((c_hat - c_true) ** 2 + (k_hat - k_true) ** 2) # costo MSE de parámetros

    loss = loss_traj + 0.5 * loss_param # costo total

    loss.backward()
    optimizer.step()

    if (it + 1) % 200 == 0:
        print(
            f"iter {it+1:4d} | "
            f"traj={loss_traj.item():.3e} | "
            f"param={loss_param.item():.3e}"
        )
```

Para evaluar podemos graficar un caso, junto también a un ajuste mediante retropropagación del gradiente como visto en la sección anterior. 

```{code-cell} ipython3

def ajuste_ck(y_obs, M, T, dt, m, x0, v0, iters=100, lr=5e-2):
    """
    Ajuste por optimización por muestra de c y k dados los datos observados y las condiciones iniciales.

    Parámetros:
    - y_obs: (T+1,) datos observados con ruido
    - M: número de observaciones iniciales usadas como input de la red
    - T: número de pasos de tiempo
    - dt: paso de tiempo
    - m: masa del oscilador
    - x0: condición inicial posición
    - v0: condición inicial velocidad
    - iters: número de iteraciones de optimización
    - lr: tasa de aprendizaje del optimizador

    Devuelve:
    - c_est: coeficiente de amortiguamiento estimado
    - k_est: constante elástica estimada
    - mse_future: error cuadrático medio en la predicción futura (después de M)
    """

    raw_c = torch.nn.Parameter(torch.tensor(0.0, device=device))
    raw_k = torch.nn.Parameter(torch.tensor(0.0, device=device))
    opt = torch.optim.Adam([raw_c, raw_k], lr=lr)

    for _ in range(iters):
        opt.zero_grad()
        c = positive(raw_c)
        k = positive(raw_k)
        x_hat = simulador_oscilador_amortiguado_batch(m, c.unsqueeze(0), k.unsqueeze(0), x0, v0, dt, T)[0]
        loss = torch.mean((x_hat - y_obs) ** 2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        c_est = positive(raw_c).item()
        k_est = positive(raw_k).item()
        x_hat = simulador_oscilador_amortiguado_batch(m, torch.tensor([c_est], device=device), torch.tensor([k_est], device=device), x0, v0, dt, T)[0]
        mse_future = torch.mean((x_hat[M:] - y_obs[M:]) ** 2).item()

    return c_est, k_est, mse_future


def graficar_un_caso(
    net, M, T, dt, m, x0, v0,
    ruido_std=0.01,
    comparar_ajuste=True,
    t_extrap_max=5.0
):
    net.eval()

    T_ext = int(t_extrap_max / dt) #definir numer de pasos extrapolados

    # --------
    # Generar un caso nuevo 
    # --------
    with torch.no_grad():
        x_partial, y_obs, c_true, k_true = sample_batch(
            batch_size=1, M=M, T=T, dt=dt, m=m, x0=x0, v0=v0, ruido_std=ruido_std
        )

        y = y_obs[0]          # (T+1,)
        c_true = c_true.item()
        k_true = k_true.item()

        # --------
        # Inferencia con la red
        # --------
        raw_c, raw_k = net(x_partial)
        c_hat = positive(raw_c)[0].item()
        k_hat = positive(raw_k)[0].item()

        x_hat_nn = simulador_oscilador_amortiguado_batch(
            m,
            torch.tensor([c_hat], device=device),
            torch.tensor([k_hat], device=device),
            x0, v0, dt, T_ext
        )[0]

    # --------
    # (Opcional) ajuste por optimización por muestra
    # --------
    if comparar_ajuste:
        c_fit, k_fit, _ = ajuste_ck(
            y, M, T, dt, m, x0, v0, iters=100
        )

        x_hat_fit = simulador_oscilador_amortiguado_batch(
            m,
            torch.tensor([c_fit], device=device),
            torch.tensor([k_fit], device=device),
            x0, v0, dt, T_ext
        )[0]

    # --------
    # Gráfico
    # --------
    with torch.no_grad():
        t_obs = torch.arange(T + 1) * dt.cpu()
        t_ext = torch.arange(T_ext + 1) * dt.cpu()


    plt.figure(figsize=(10, 5))

    plt.plot(t_obs, y.cpu(), "k.", markersize=2, alpha=0.4, label="Observaciones")
    plt.plot(t_ext, x_hat_nn.cpu(), "r", linewidth=2, label="Predicción (NN + DP)")

    if comparar_ajuste:
        plt.plot(t_ext, x_hat_fit.cpu(), "b--", linewidth=2, label="Predicción (ajuste por muestra)")

    plt.axvline(t_obs[M], color="gray", linestyle=":", label="Fin observación")
    plt.axvline(t_obs[-1], color="k", linestyle="--", label="Fin dominio entrenamiento")
    plt.axvspan(
        t_obs[-1],
        t_ext[-1],
        color="orange",
        alpha=0.15,
        label="Región de extrapolación"
    )

    plt.title(
        "Inferencia de parámetros con Física Diferenciable + Red Neuronal\n"
        f"True: c={c_true:.3f}, k={k_true:.3f} | "
        f"NN: c={c_hat:.3f}, k={k_hat:.3f}"
        + (f" | Fit: c={c_fit:.3f}, k={k_fit:.3f}" if comparar_ajuste else "")
    )

    plt.xlabel("Tiempo")
    plt.ylabel("Posición x(t)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

graficar_un_caso(
    net,
    M=M,
    T=T,
    dt=dt,
    m=m_true,
    x0=x0,
    v0=v0,
    ruido_std=0.01,
    comparar_ajuste=True,
    t_extrap_max=5.0
)
```

En el título del gráfico, vemos los parámetros verdaderos, como los obtenidos por la red neuronal y aquellos obtenidos por el ajuste de los datos por retropropagación, utilizando la diferenciabilidad de nuestro simulador. Incrementando la cantidad de iteraciones se puede lograr una aproximación excelente y vemos que la solución que produce la red neuronal es consistente con las ecuaciones diferenciales propuestas por el simulador. A su vez, vemos como en la región de extrapolación la solución encontrada y propuesta por


## Discusión final

Agregar discusión.

```{bibliography}
:style: unsrt
:filter: docname in docnames
```
