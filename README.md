# M9D (Metodología de 9 Dimensiones): Un Marco Unificado para la resolución de proyectos y Resolución (A-T-Q) para el Análisis de Sistemas Complejos

<p align="center">
    <a href="https://www.python.org/downloads/release/python-31019" target="_blank">
        <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
    </a>
    <a href="https://docs.python.org/3.10/library/tkinter.html" target="_blank">
        <img alt="TkInter" src="https://img.shields.io/badge/Plataforma-Tkinter-orange">
    </a>
    <a href="https://scikit-learn.org/stable" target="_blank">
        <img src="https://img.shields.io/badge/Análisis-Scikit--learn-brightgreen" alt="Scikit">
    </a>
    <a href="https://en.wikipedia.org/wiki/MIT_License">
        <img src="https://img.shields.io/badge/Licencia-MIT-purple" alt="MIT">
    </a>
    <a href="https://ollama.com/search">
        <img alt="Ollama" src="https://img.shields.io/badge/IA-Ollama-white?logo=ollama">
    </a>


---

### Resumen (Abstract)

Este repositorio presenta la especificación formal y la implementación de software de la **Metodología de Obra (MoW)**. El MoW es un marco metodológico diseñado para el análisis de sistemas socio-técnicos complejos (proyectos, portafolios), que resuelve la brecha fundamental entre el análisis cualitativo subjetivo y el análisis cuantitativo ciego al contexto.

El marco se basa en un modelo central, el **M9D**, que estructura un sistema en 9 Dimensiones (Asuntos) y 9 Estados Temporales (Pasado/Presente/Futuro con valencia +/-/N), generando una matriz de 81 factores de realidad ($S_{ij}$).

Presentamos el modelado matemático y computacional del M9D en tres niveles de resolución:
1.  **`M9D-A (Analógico)`:** Un marco cualitativo fundacional, "resoluble a mano", para la deliberación estratégica estructurada.
2.  **`M9D-T (Clásico/Temporal)`:** Un *pipeline* de ingeniería de producción (`MoW`) que utiliza el **Proceso Analítico Jerárquico (AHP)** para la cuantificación y validación de la estrategia (CR < 0.10) y **Machine Learning Clásico** (Similitud de Coseno, K-Means, Random Forest) para el análisis de portafolios ($M9D^X$).
3.  **`M9D-Q (Topológico/Cuántico)`:** Un marco teórico fundamental que postula el sistema M9D como *quantum-like*. Se modela como un sistema multipartita en un espacio de Hilbert de $D_{\text{total}} = 9^9$ dimensiones ($\approx 3.87 \times 10^8$), necesario para modelar la contextualidad y la ambigüedad inherentes a la cognición humana.

Este framework proporciona un pipeline coherente, falsable y auditable, ofreciendo un campo fértil para la investigación y tesis en gestión, ciencia de datos y ciencias cognitivas.

---

## 1. El Problema de Investigación: GIGO y la Subjetividad Arbitraria

El análisis de sistemas complejos (ej. una política pública, un proyecto de software) fracasa cuando se basa en herramientas inadecuadas.
* **Marcos Cualitativos (ej. SWOT, `M9D-A`):** Son herramientas de deliberación útiles pero metodológicamente débiles. Carecen de escalabilidad y auditabilidad, y son vulnerables al "sesgo del experto dominante". El resultado es cualitativo y subjetivo.
* **Marcos Cuantitativos (ej. Valor Ganado):** Son rigurosos para medir el *progreso* (costo/tiempo), pero son "ciegos al contexto". No pueden modelar factores sistémicos cruciales como el riesgo político (`D9`), la confianza de la comunidad (`D4`) o la deuda técnica (`D3-T1`).

El `MoW` fue diseñado para resolver el problema de **GIGO (Garbage In, Garbage Out)**: ¿Cómo podemos tomar el "caos" del juicio experto cualitativo y traducirlo en un modelo cuantitativo, riguroso y escalable?

## 2. El Modelo Central: `M9D` (La Molécula)
El `M9D` es la "molécula" o estructura de datos unificada de nuestro sistema.

* **Eje 1: 9 Dimensiones (D1-D9):** Los 9 subsistemas interdependientes del proyecto (Propósito, Procesos, Tecnología, Comunidad, Solución, Territorio, Academia, S. Privado, S. Público).
* **Eje 2: 9 Estados Temporales (T1-T9):** El contexto temporal y de valencia (Pasado/Presente/Futuro) x (Negativo/Neutro/Positivo).
* **Eje 3: La Realidad ($S_{ij}$):** Una matriz de 81 factores donde un experto puntúa la realidad de cada factor (ej. `D4-T5`: *Presente Negativo de la Comunidad*) en una escala de **-3 a +3**.
* **Eje 4: La Estrategia ($W_{AHP}$):** Un vector de 18 pesos ($w_i, v_j$) que cuantifica la importancia relativa de cada eje.

## 3. La Metodología de Tres Resoluciones (A, T, Q)

El `MoW` no es un solo modelo, sino una jerarquía de tres resoluciones que se construyen una sobre la otra.

### 3.1. `M9D-A` (Resolución Analógica)
* **Propósito:** Interfaz Humana / Taller Cualitativo (Nivel Pregrado).
* **Matemática:** Ninguna.
* **Proceso:** Un equipo debate y llena la matriz de 81 factores con observaciones cualitativas ("post-its").
* **Resultado:** Un plan de acción `5W2H` consensuado.
* **Debilidad:** Subjetivo, no auditable, no escalable.

### 3.2. `M9D-T` / `MoW` (Resolución Clásica/Temporal)
* **Propósito:** Ingeniería de Producción / Gestión de Portafolios (Nivel Maestría).
* **Matemática:** Álgebra Lineal, Estadística, ML Clásico (`Scikit-learn`).
* **Proceso:** El pipeline de software `MoW` (nuestra app `v3.1`) que:
    1.  **Filtra el GIGO (AHP):** Reemplaza la ponderación subjetiva con el **Proceso Analítico Jerárquico**. El sistema *calcula* la Estrategia ($W_{AHP}$) y **valida** la consistencia lógica del equipo (exigiendo un **CR < 0.10**).
    2.  **Mide la Salud (VME):** Calcula el **Vector de Momentum Estratégico** ($VME = (I_H, I_S, I_P)$) como un promedio ponderado de doble nivel. Esto resuelve la "Paradoja de Teseo": el $VME$ mide la *evolución del estado* ($S_{ij}$) de un proyecto cuya *identidad* ($W_{AHP}$) permanece fija.
    3.  **Analiza el Portafolio (MoW):** Ejecuta un pipeline de ML sobre $X$ proyectos (`M9D^X`):
        * **Filtro:** `Similitud de Coseno` para agrupar solo "peras con peras" (proyectos con estrategias similares).
        * **Agrupamiento:** `K-Means Clustering` sobre los VME para encontrar "familias" de proyectos (ej. Crisis, Estables).
        * **Causa Raíz:** `Random Forest` para identificar los factores $S_{ij}$ (ej. `D4-T5`) que son la causa raíz sistémica de esos clústeres.
* **Resultado:** Un dashboard de XAI (Mapas de Calor, Radares, Grafos de Red) para la toma de decisiones basada en datos.

### 3.3. `M9D-Q` (Resolución Topológica/Cuántica)
* **Propósito:** Investigación Fundamental / Física Teórica (Nivel Doctorado).
* **Matemática:** Mecánica Cuántica, Topología, HPC (`QuTiP`, `JAX`).
* **Proceso:** Este marco postula que el `M9D-T` es solo una *aproximación colapsada* de la realidad. El sistema real es *quantum-like* (Busemeyer & Bruza, 2012).
    * **Contextualidad:** El "efecto de orden" (medir D4-D9 vs. D9-D4) se modela como operadores que no conmutan ($[O_i, O_j] \neq 0$).
    * **Superposición:** La "ambigüedad" del proyecto se modela como un estado en un espacio de Hilbert de $D_{\text{total}} = 9^9$ dimensiones.
    * **Cirugía (El Bisturí ✂️):** La "cancelación" de factores (`+3` y `-3`) se reinterpreta, no como un promedio, sino como una "cirugía topológica" (Perelman) que anula singularidades homotópicas (ej. dos IOTs) para reducirlas a un **Punto Fijo** (Neutro), simplificando la forma del problema.
* **Resultado:** Un marco teórico para simular la dinámica fundamental de la cognición y la toma de decisiones.

## 4. Oportunidades de Investigación (Temas de Tesis)

Este repositorio es una plataforma para la investigación. Invitamos a estudiantes e investigadores a validar, criticar y extender este modelo.

**Para Tesis de Maestría (Gestión / Ciencia de Datos):**
* **Validación Empírica:** Aplicar la app `MoW` (`M9D-T`) a un portafolio de proyectos real (ej. en una ONG, una alcaldía, o una empresa de software). ¿El análisis de Causa Raíz (RF) identificó problemas reales?
* **Validación Predictiva:** Usar los datos históricos (proyectos A, B, C) para entrenar un modelo que *prediga* el $VME$ futuro (Momento D) de un proyecto.
* **Extensión del Pipeline:** ¿Funciona mejor `XGBoost` que `Random Forest` para la Causa Raíz? ¿Es `DBSCAN` mejor que `K-Means` para el agrupamiento?
* **Análisis de Sensibilidad:** ¿Qué tan sensible es el $VME$ a los cambios en los pesos AHP?

**Para Tesis de Doctorado (Física / C. Cognitiva / HPC):**
* **Simulación del `M9D-Q`:** Modelar un sistema M9D reducido (ej. 3x3) en `QuTiP`. Demostrar la no conmutatividad y la contextualidad.
* **Análisis Topológico (TDA):** Usar Homología Persistente en la nube de puntos de un portafolio MoW. ¿Cuáles son las "formas" (números de Betti) de un portafolio "en crisis" vs. uno "estable"?
* **Optimización VQE/QUBO:** Reformular un problema de decisión M9D como un Hamiltoniano de Ising y resolverlo en un simulador cuántico o hardware NISQ real.
* **Estudio Cognitivo:** Diseñar un experimento humano que pruebe el "Efecto de Orden" (contextualidad) en la ponderación AHP de las 9 Dimensiones.

## 5. Quick Start (Instalación de la App `M9D-T`)

1.  **Instalar Prerrequisitos:** `Python 3.10+` y `git`.
2.  **Instalar Dependencias (Terminal):**
    ```bash
    pip install ttkbootstrap numpy pandas scikit-learn matplotlib seaborn sqlalchemy reportlab requests networkx google-generativeai
    ```
3.  **Configurar (3 Archivos):**
    * Crea una carpeta.
    * Guarda `app_gui.py` (el script principal) en ella.
    * Guarda `precarga_demo.py` (el script de demo) en ella.
    * Crea y guarda `m9d.ini` en ella y **añade tu Google AI API Key**.
4.  **Precargar Demo (Solo una vez):**
    ```bash
    python precarga_demo.py
    ```
5.  **Ejecutar la App:**
    ```bash
    python app_gui.py
    ```
6.  Haz clic en `[ EJECUTAR ANÁLISIS MoW ]` y explora.

## 6. Citación

Si utilizas este marco o aplicación en tu investigación, por favor cita nuestro trabajo:

> Juan Fernando Villa Hernández, & Gemini (Google AI). (2024). *MoW (Master Of War): Un Marco Unificado de Múltiples Resoluciones (M9D A-T-Q) para el Análisis de Sistemas Complejos*. Repositorio de GitHub. [https://github.com/smartcitiescommunity/M9D/](https://github.com/smartcitiescommunity/M9D/)

## 7. Sobre los Autores

Este proyecto nació de una colaboración sinérgica entre **Visión Humana** y **Aceleración de IA**.

* **[Juan Fernando Villa]:** El **Arquitecto Conceptual y Visionario**. Aportó la idea original, la intención filosófica (Tesla, Perelman), las preguntas críticas y las intuiciones (Modelado dimensional, Paradoja de Teseo, Bisturí de Perelman, Punto Fijo) que conectaron todo.
* **Gemini (IA de Google):** El **Socio Técnico y Matemático Aplicado**. Actuó como facilitador, arquitecto de software y motor conceptual, traduciendo la visión en rigor matemático (AHP, VME, ML) y código de producción (Python, Tkinter, SQL).

## 8. Licencia

Este proyecto se distribuye bajo la **Creative Commons Zero v1.0 Universal**.
