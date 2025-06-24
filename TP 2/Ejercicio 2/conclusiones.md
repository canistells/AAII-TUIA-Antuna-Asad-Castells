# Aprendizaje Automático II - Trabajo Práctico N°2

- Antuña, Franco (A-4637/1)
- Asad, Gonzalo (A-4595/1)
- Castells, Sergio (C-7334/2)

----------------

# Problema 2 - Flappy Bird

## Ingeniería de Características

### Estado Crudo del Entorno

* 'player y position': Posición Y del centro del pájaro.
* 'players velocity': Velocidad Y actual del pájaro.
* 'next pipe distance to player': Distancia al pájaro en X de la próxima cañería.
* 'next pipe top y position': Posición Y de la cañería superior.
* 'next pipe bottom y position': Posición Y de la cañería inferior.

Si bien el juego incluye otros tres estados, referentes a dos cañerías adelante:

* 'next next pipe distance to player'
* 'next next pipe top y position'
* 'next next pipe bottom y position'

Los mismos se descartan para el diseño de la Q-Table con el objetivo de reducir la cantidad de estados posibles.

### Características del Estado Procesadas y Discretizadas

<p align="center">Estado crudo → Tupla de 5 números enteros.</p>

1. _¿Dónde está el pájaro verticalmente en la pantalla?_

**Idea:** ¿Está el pájaro en la parte superior, en el centro o en la parte inferior de la pantalla?

**Simplificación:** En lugar de la posición exacta en píxeles, dividimos la altura del campo en unas pocas zonas, por ejemplo, 20 zonas. Así, sólo necesitamos saber si la bola está en la "zona 1 (muy arriba)", "zona 10 (centro)" o "zona 20 (muy bajo)".

**Importancia:** Nos dice si el pájaro está alineado verticalmente con el espacio entre las cañerías o no.

2. _¿Cómo me estoy moviendo ahora mismo?_

**Idea:** ¿El pájaro está subiendo o bajando rápidamente, lentamente o está relativamente quieto?

**Simplificación:** En lugar de la velocidad exacta en píxeles por segundo, dividimos la velocidad en unas pocas, por ejemplo, 10 zonas. Así, solo necesitamos saber si el pájaro "asciende lentamente", "desciende lentamente", "desciende rápidamente" o "está quieto".

**Importancia:** Ayuda a coordinar el movimiento actual con el movimiento deseado.

3. _¿Qué tan lejos está la próxima cañería horizontalmente?_

**Idea:** ¿La cañería está muy lejos del pájaro? ¿Está muy cerca?

**Simplificación:** Dividimos el ancho del campo en varias secciones, por ejemplo, 10 secciones. Así, sabemos si la cañería está "zona 1 (muy a la izquierda)", "zona 5 (en el centro de la pantalla)" o "zona 10 (muy a la derecha)".

**Importancia:** Da una idea de cuán próximo está el siguiente desafío y si hay tiempo suficiente de reacción.

4. _¿Qué tan lejos está el fin de la cañería superior verticalmente?_

**Idea:** ¿El final de la cañería superior está muy lejos del pájaro? ¿Está muy cerca?

**Simplificación:** Dividimos el alto del campo en varias secciones, por ejemplo, 20 secciones. Así, sabemos si el final de la cañería superior está "zona 1 (muy arriba)", "zona 10 (en el centro de la pantalla)" o "zona 20 (muy abajo)".

**Importancia:** Da una idea de cuán próximo está el pájaro de la abertura entre caños, si un salto lo compromete y si hay tiempo suficiente de reacción.

5. _¿Qué tan lejos está el fin de la cañería inferior verticalmente?_

**Idea:** ¿El final de la cañería inferior está muy lejos del pájaro? ¿Está muy cerca?

**Simplificación:** Dividimos el alto del campo en varias secciones, por ejemplo, 20 secciones. Así, sabemos si el final de la cañería superior está "zona 1 (muy arriba)", "zona 10 (en el centro de la pantalla)" o "zona 20 (muy abajo)".

**Importancia:** Da una idea de cuán próximo está el pájaro de la abertura entre caños, si un salto lo compromete y si hay tiempo suficiente de reacción.

### Ejemplo

* "Pájaro un poco arriba en la pantalla" (Componente 7)
* "Pájaro cayendo muy rápidamente" (Componente 10)
* "Próxima cañería no muy lejos del pájaro" (Componente 8)
* "Final de la cañería superior un poco arriba del centro de la pantalla" (Componente 7)
* "Final de la cañería inferior al centro de la pantalla" (Componente 10)

Esta combinación de respuestas simplificadas ((un_poco_arriba, muy_rápidamente, no_muy_lejos, un_poco_arriba, al_centro)) forma el estado discreto que el agente usa para buscar en su "libro de jugadas" (la Q-Table) cuál es la mejor acción a tomar (aletear o no aletear).

## Análisis de Desarrollo

El entrenamiento del Q-Agent requirió una discretización del espacio de estados. Inicialmente, se experimentó con un número importante de bins para la discretización de las variables de estado. Sin embargo, se observó que esto conducía a un espacio de estados excesivamente grande, lo que dificultaba significativamente el proceso de entrenamiento. Tras varias iteraciones, se optó por una discretización con cinco estados. Esta decisión se tomó en base a la necesidad de mantener un tamaño manejable para la Q-Table, optimizando así la relación entre desempeño y complejidad del modelo.

La fase inicial de entrenamiento del Q-Agent fue lenta. Durante los primeros 2500 episodios, no se observaron mejoras significativas en el rendimiento del agente. Sin embargo, a partir de allí, la mejora de las métricas fue notoria. Esto indica que la Q-Table comenzó a converger efectivamente una vez que se exploró una porción suficiente del espacio de estados.

El desempeño obtenido con el Q-Agent fue muy bueno, alcanzándose un promedio de 150 puntos en las pruebas. Por otro lado se observó lo siguiente: el agente tendía a perder la partida en situaciones que demandaban un ascenso vertical considerable. Esto podría deberse a que estas transiciones de estado no fueron exploradas o valoradas adecuadamente en la Q-Table.

El valor del hiperparámetro épsilon fue determinado mediante un proceso empírico de prueba y error. Valores muy altos de épsilon resultaban en una cantidad excesiva de acciones aleatorias. 

Para el caso del agente basado en Deep Q-Learning (NN-Agent) se plantearon dos enfoques para la red neuronal: regresión y clasificación.

La implementación basada en un modelo de clasificación no arrojó métricas de entrenamiento satisfactorias. Esto pudo deberse a un error en el diseño de la arquitectura.

Por otro lado, el modelo de regresión mostró un buen desempeño, con buenas métricas de entrenamiento (val_mse de aproximadamente 0.3).

<p><img src="https://raw.githubusercontent.com/canistells/AAII-TUIA-Antuna-Asad-Castells/refs/heads/main/TP%202/Ejercicio%202/entrenamiento_resultados.png" alt="Métricas de Entrenamiento."></p> 

La arquitectura de la red neuronal fue diseñada con una capa de entrada cuyo tamaño corresponde a la cantidad de estados de la Q-Table y una capa de salida con tantas neuronas como acciones posibles (dos, en este caso: volar/no volar). Además se implementaron capas intermedias. Esta configuración permitió que la red aprendiera a predecir los valores Q para cada acción dado un estado de entrada.

## Conclusiones

Aunque el NN-Agent mostró un buen rendimiento general, los resultados obtenidos en las pruebas tan buenos como los alcanzados por el Q-Agent. Se observó que, en promedio, el NN-Agent obtuvo xx puntos, mientras que el Q-Agent logró yy puntos. Sin embargo, se observó que el Q-Agent tuvo rachas esporádicas de puntaje significativamente más alto que las observadas en el NN-Agent. Esto sugiere que, si bien el NN-Agent pudo aprender una política general, el Q-Agent podría haber logrado una optimización más fina en ciertos estados críticos o haber explorado combinaciones de estados-acción que la red neuronal no representó con la misma fidelidad.

En cuanto al rendimiento, el Q-Agent mostró a una velocidad cercana a 30 FPS. En contraste, el NN-Agent presentó un rendimiento más lento, principalmente debido al costo computacional asociado al proceso de inferencia de la red neuronal en cada paso de tiempo.
