# Uso de inteligencia artificial generativa

Este documento recoge el uso de inteligencia artificial generativa realizado en
el proyecto, de acuerdo con las condiciones indicadas en el enunciado del
trabajo.

## Sistema utilizado

Se ha utilizado Codex, basado en GPT-5, integrado en el entorno de desarrollo.
El sistema se ha usado como asistente de programacion, revision y documentacion.

No se ha usado IA generativa para sustituir la comprension del trabajo. Los
integrantes deben revisar, ejecutar y entender el codigo antes de la defensa.

## Usos realizados

La IA generativa se ha usado para las siguientes tareas:

- Leer y resumir el enunciado del trabajo.
- Proponer una arquitectura inicial del proyecto.
- Generar una primera version del motor de Otelo.
- Generar una implementacion propia de MCTS con UCT.
- Generar agentes de referencia: aleatorio, voraz, heuristico, UCT y UCTNN.
- Crear scripts para autojuego, entrenamiento, torneos e interfaz de texto.
- Implementar una red neuronal de valor con NumPy.
- Preparar pruebas unitarias.
- Ejecutar experimentos reproducibles.
- Actualizar tablas y resultados de la documentacion.
- Preparar una version inicial de la memoria en LaTeX.

## Prompts principales y consultas realizadas

Durante el desarrollo se realizaron consultas de distinto nivel de detalle. A
continuacion se recogen ejemplos representativos de las instrucciones usadas o
de las consultas tecnicas realizadas:

- "Analiza el enunciado del trabajo de Otelo e identifica los requisitos
  obligatorios para la convocatoria de junio".
- "Propón una arquitectura de proyecto en Python que separe reglas del juego,
  agentes, entrenamiento, experimentos y documentacion".
- "Diseña una representacion del tablero adecuada para Otelo y para entrenar una
  red neuronal de valor".
- "Implementa la generacion de movimientos legales de Otelo comprobando las ocho
  direcciones de captura".
- "Implementa una version propia de Monte Carlo Tree Search con seleccion UCT
  para un juego de suma cero".
- "Explica como debe cambiar MCTS cuando se usa una red neuronal en lugar de la
  politica por defecto basada en rollouts".
- "Crea agentes de referencia para comparar el rendimiento: aleatorio, voraz,
  heuristico, UCT y UCT con red neuronal".
- "Genera un script de autojuego que guarde estados intermedios y los etiquete
  con victoria, empate o derrota desde la perspectiva del jugador activo".
- "Diseña una red neuronal sencilla de valor con salida en el intervalo [-1, 1]
  y entrenable con NumPy".
- "Prepara un script de entrenamiento que cargue el dataset, entrene la red y
  guarde los pesos en un archivo .npz".
- "Crea torneos reproducibles entre agentes alternando colores para reducir el
  sesgo de mover primero".
- "Ejecuta un experimento completo con autojuego, entrenamiento y torneos, y
  guarda los resultados en Markdown y JSON".
- "Revisa si los resultados son coherentes y si la red neuronal realmente aporta
  mejora frente a los agentes basicos".
- "Amplia el dataset de autojuego para mejorar la calidad de la red y actualiza
  las tablas de resultados".
- "Redacta una memoria en formato IEEE con introduccion, preliminares,
  implementacion, experimentos, conclusiones y bibliografia".
- "Incluye en la memoria diagramas explicativos de las fases de UCT, la red de
  valor y el flujo de entrenamiento".
- "Explica la diferencia entre jugar contra UCT con red y UCT sin red".
- "Explica que contiene el dataset .npz y que contiene el modelo entrenado".

Ademas de esos prompts principales, se hicieron preguntas de aclaracion sobre el
papel de algunos ficheros, el dataset o el modelo entrenado.

## Revision humana necesaria

Antes de entregar el trabajo, los integrantes han comprobado personalmente:

- Que las reglas de Otelo implementadas son correctas.
- Que UCT selecciona movimientos legales.
- Que la red se entrena con los datos generados por autojuego.
- Que el agente UCTNN usa el modelo entrenado para evaluar posiciones.
- Que los resultados de los experimentos se pueden reproducir.
- Que ambos integrantes saben explicar el codigo durante la defensa.

## Limitaciones del uso de IA

La IA generativa puede proponer codigo incorrecto o explicaciones incompletas.
Por ese motivo, el proyecto se ha validado mediante:

- pruebas unitarias;
- ejecucion de partidas automaticas;
- entrenamiento real de la red;
- torneos entre agentes;
- compilacion de la memoria en PDF.

La responsabilidad final sobre el codigo, los resultados y la defensa es de los
autores del trabajo.
