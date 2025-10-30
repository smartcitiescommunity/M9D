# 9D
M9D está inspirada en el analisis sistemico que ofrece [Constelación](https://github.com/smartcitiescommunity/Constelation), el cual es fundamental en el analisis y la toma de decisiones. 

# M9D es un [Modelo, MEtodo y Metodología] dependiendo de como sea usado (M9D^X=MoW)
Un sistema de análisis de portafolio, de código abierto, que convierte el juicio experto en estrategia matemática y acción de ML.

<p align="center">
    <a href="https://www.python.org/downloads/release/python-31019" target="_blank">
        <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
    </a>
    <a href="https://docs.python.org/3.10/library/tkinter.html" target="_blank">
        <img alt="TkInter" src="https://img.shields.io/badge/Plataforma-Tkinter-orange">
    </a>
    <a href=https://scikit-learn.org/stable" target="_blank">
        <img src="(https://img.shields.io/badge/build-passing-brightgreen" alt="Scikit">
    </a>
    <a href="https://en.wikipedia.org/wiki/MIT_License">
        <img src="https://img.shields.io/badge/Licencia-MIT-purple" alt="MIT">
    </a>
    <a href="https://ollama.com/search">
        <img alt="Ollama" src="https://img.shields.io/badge/IA-Ollama-white?logo=ollama">
    </a>
</p>

---

> "La gestión de proyectos moderna es un caos de intuición y subjetividad arbitraria. Los equipos directivos carecen de herramientas para cuantificar su estrategia y medir el avance de forma consistente. **M9D** es un sistema para cambiar eso."

**M9D** es una plataforma de análisis de portafolio que traduce la complejidad del juicio experto en un modelo matemático riguroso, auditable y accionable.

Este no es solo un tablero de control. Es un "sistema operativo" para la toma de decisiones estratégicas que te permite:

1.  **Cuantificar tu Estrategia:** Traduce el debate cualitativo (ej. "¿qué es más importante?") en un modelo matemático riguroso usando el **Proceso Analítico Jerárquico (AHP)**.
2.  **Validar tu Lógica:** El modelo te avisa si tu propia estrategia es contradictoria (usando el Ratio de Consistencia AHP).
3.  **Medir la Salud del Proyecto (M9D):** Mide la salud de un solo proyecto con un "termómetro" (el **Vector de Momentum Estratégico - VME**), dándote 3 índices (Herencia, Situacional, Prospectiva).
4.  **Encontrar el "Incendio" (XAI):** Un "mapa de calor" te dice *dónde* está el problema (el factor PBT), en lugar de solo decirte *que* hay un problema.
5.  **Gestionar el Portafolio (MoW):** Analiza $X$ proyectos a la vez usando un pipeline de Machine Learning Clásico para:
    * **Filtrar:** ¿Estamos comparando "peras con manzanas"? (Similitud de Coseno).
    * **Agrupar:** ¿Qué "familias" de proyectos tenemos? (K-Means Clustering).
    * **Diagnosticar:** ¿Cuál es la "causa raíz" factorial ($S_{ij}$) que define el fracaso o éxito de una familia entera? (Random Forest).
6.  **Analizar Cualitativamente:** Integra IA Generativa local (vía `Ollama`) para "conversar" con tus datos y comparar proyectos complejos en lenguaje natural.

---

## 🏛️ Filosofía: Conocimiento Libre

Este proyecto se libera al mundo bajo la **Licencia MIT**.

Es una herramienta de conocimiento libre para la humanidad. Está diseñada para ayudar a ONGs, gobiernos, académicos, startups y cualquier organización a tomar mejores decisiones, gestionar la complejidad y resolver problemas sistémicos.

## 🚀 El "Stack" Tecnológico

* **GUI:** Python `Tkinter` con `ttkbootstrap` para una interfaz moderna.
* **Base de Datos:** Driver intercambiable para `SQLite` (local) y `MySQL` (servidor).
* **Motor Matemático (AHP/VME):** `Numpy` y `Pandas`.
* **Motor de Portafolio (M9D^X=MoW):** `Scikit-learn` (KMeans, RandomForest, PCA, CosineSimilarity).
* **Motor de IA Cualitativa:** `Ollama` (conectado vía `requests`).
* **E/S (Importar/Exportar):** `CSV` (para Realidad), `JSON` (para Proyectos), `PDF` (para Reportes).
* **Producción:** `threading` y `queue` para una GUI que nunca se congela.

## ⚡ Quick Start (Puesta en Marcha)

Esta es una aplicación de escritorio.

### Prerrequisitos

1.  **Python 3.10+** instalado.
2.  **Ollama** instalado y corriendo en segundo plano. (Ej. `ollama run llama3:8b`)

### Instalación

1.  Clona este repositorio:
    ```bash
    git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
    cd TU_REPOSITORIO
    ```
2.  (Recomendado) Crea un entorno virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # (o .\venv\Scripts\activate en Windows)
    ```
3.  Instala las dependencias (asegúrate de crear un archivo `requirements.txt` con el contenido de las `pip install...`):
    ```bash
    pip install -r requirements.txt
    ```

### Ejecución

1.  Asegúrate de que Ollama esté corriendo en otra terminal.
2.  Ejecuta la aplicación (asumiendo que la guardaste como `mow_app_v2.py`):
    ```bash
    python mow_app_v2.py
    ```
3.  ¡Listo! Sigue el flujo de trabajo:
    * **Paso 1:** Ve a la Pestaña `(3) Estrategias (AHP)` -> "Crear Nueva Estrategia...".
    * **Paso 2:** Llena los 45 sliders y guarda una estrategia válida (CR < 10%).
    * **Paso 3:** Ve al `Panel de Control` -> "Crear Nuevo Proyecto...".
    * **Paso 4:** Ve a la Pestaña `(2) Proyecto (M9D)` -> "Importar Realidad (CSV)...".
    * **Paso 5:** Repite 3 y 4 para varios proyectos.
    * **Paso 6:** Ve al `Panel de Control` -> "EJECUTAR ANÁLISIS MoW".
    * **Paso 7:** Revisa los resultados en la Pestaña `(1) Portafolio (MoW)`.

---

## 🧠 La Metodología (El "Cerebro")

### Nivel 1: El Proyecto (M9D)
El **M9D** es el modelo para un solo proyecto. Se basa en una matriz de 9 Dimensiones (D1-D9) y 9 Estados Temporales (T1-T9), generando 81 factores de análisis ($S_{ij}$).

Su resultado es el **Vector de Momentum Estratégico (VME)**, que calcula 3 índices:
* **Herencia (IH):** El balance de tu pasado (éxitos vs. fracasos).
* **Situacional (IS):** El balance de tu presente (fortalezas vs. debilidades).
* **Prospectiva (IP):** El balance de tu futuro (visión vs. riesgos).

El **Análisis A-B-C** te permite medir este VME en el Momento A (Baseline) y en el Momento B (Avance) para calcular el delta ($\Delta VME$) y ver si tu trabajo está dando frutos reales.

### Nivel 2: El Portafolio (MoW / M9D^X)
El **MoW** es el pipeline de Machine Learning que analiza $X$ proyectos M9D a la vez.

1.  **Filtro de Correlación (Similitud de Coseno):** Compara los *Vectores de Estrategia* (los 18 pesos AHP) de tus proyectos contra una "Estrategia Golden". Esto responde: "¿Estamos comparando peras con manzanas?".
2.  **Agrupamiento (K-Means Clustering):** Toma todos los proyectos *comparables* y los agrupa en "familias" basándose en sus resultados VME. Esto responde: "¿Qué tipos de proyectos tenemos? (ej. 'Crisis', 'Estables', 'Oportunidades')".
3.  **Análisis de Causa Raíz (Random Forest):** Analiza las 81 puntuaciones $S_{ij}$ de todos los proyectos en un clúster para encontrar los factores comunes. Esto responde: "¿*Por qué* la familia 'Crisis' está en crisis? (ej. "Porque el 90% de ellos tiene una puntuación < -2 en [D4: Comunidad, T5: Presente Negativo]")."

## 🤝 Cómo Contribuir

¡Este proyecto está vivo! Eres bienvenido a contribuir.

* **Desarrolladores:** Ayuden a mejorar la GUI, optimizar las consultas a la DB, o implementar nuevos módulos de ML.
* **Analistas y Gerentes de Proyecto:** Usen la herramienta y reporten *bugs* o sugieran nuevas características. ¿Qué echan en falta en su día a día?
* **Académicos y Científicos:** Tomen el modelo (es 100% falsable) y valídenlo. Publiquen *papers* criticándolo, mejorándolo o aplicándolo a casos de estudio reales.
* **Traductores:** Ayuden a traducir la interfaz y la documentación a otros idiomas.

## ❤️ Sobre los Autores

Este proyecto nació de una colaboración sinérgica entre **Visión Humana** y **Aceleración de IA**.

* **Juan Fernando Villa Hernández:** El Visionario y Gestor del Proyecto. Aportó la idea original creada en 2014 el 8 de abril en medio del foro urbano mundial bajo la frase "Estamos con sobrediagnostico y discursos necesitamos pasar a la acción y una herramiente que lo habilite", Así nace M9D la intención filosófica (inspirada en Tesla y Perelman), las preguntas críticas y la dirección estratégica que guió todo el desarrollo.
* **Gemini (IA de Google):** El Socio Técnico. Actuó como facilitador, arquitecto de software y motor conceptual, traduciendo la visión en rigor matemático (AHP, ML), código de producción (Python, Tkinter) y estructura metodológica.

## 📜 Licencia

Este proyecto se distribuye bajo la **Licencia MIT**.
