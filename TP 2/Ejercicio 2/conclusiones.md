# Aprendizaje Automático II - Trabajo Práctico N°2

- Antuña, Franco (A-4637/1)
- Asad, Gonzalo (A-4595/1)
- Castells, Sergio (C-7334/2)

----------------

# Problema 2 - Flappy Bird

## Ingeniería de Características

### Estado Crudo del Entorno:

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

### Características del Estado Procesadas y Discretizadas:



1. _¿Dónde está el pájaro verticalmente en la pantalla?_

**Idea:** ¿Está el pájaro en la parte superior, en el centro o en la parte inferior de la pantalla?

**Simplificación:** En lugar de la posición exacta en píxeles, dividimos la altura del campo en unas pocas zonas. Por ejemplo, 20 zonas. Así sólo necesitamos saber si la bola está en la "zona 1 (muy arriba)", "zona 10 (centro)" o "zona 20 (muy bajo)".

**Importancia:** Me dice si está alineado verticalmente con el espacio entre las cañerías o no.

2. _¿Cómo me estoy moviendo ahora mismo?_

**Idea:** ¿El pájaro está subiendo o bajando rápidamente, lentamente o está relativamente quieto?

**Simplificación:** En lugar de la velocidad exacta en píxeles por segundo, dividimos la velocidad en unas pocas. Por ejemplo, 10 zonas. Así solo necesitamos saber si el pájaro "asciende lentamente", "desciende lentamente", "desciende rápidamente" o "está quieto".

**Importancia:** Ayuda a coordinar el movimiento actual con el movimiento deseado.

3. _¿Qué tan lejos está la próxima cañería horizontalmente?_

**Idea:** ¿La cañería está muy lejos del pájaro? ¿Está muy cerca?

**Simplificación:** Dividimos el ancho del campo en varias secciones. Por ejemplo, 10 secciones. Así, sabemos si la cañería está "zona 1 (muy a la izquierda)", "zona 5 (en el centro de la pantalla)" o "zona 10 (muy a la derecha)".

**Importancia:** Da una idea de cuán próximo está el siguiente desafío y si hay tiempo suficiente de reacción.

4. _¿Qué tan lejos está el fin de la cañería superior verticalmente?_

**Idea:** ¿El final de la cañería está muy lejos del pájaro? ¿Está muy cerca?

**Simplificación:** Dividimos el alto del campo en varias secciones. Por ejemplo, 20 secciones. Así, sabemos si el final de la cañería superior está "zona 1 (muy arriba)", "zona 10 (en el centro de la pantalla)" o "zona 20 (muy abajo)".

**Importancia:** Da una idea de cuán próximo está pájaro de la abertura entre caños, si un salto lo compromete y si hay tiempo suficiente de reacción.

5. _¿Qué tan lejos está el fin de la cañería inferior verticalmente?_

**Idea:** ¿El final de la cañería está muy lejos del pájaro? ¿Está muy cerca?

**Simplificación:** Dividimos el alto del campo en varias secciones. Por ejemplo, 20 secciones. Así, sabemos si el final de la cañería superior está "zona 1 (muy arriba)", "zona 10 (en el centro de la pantalla)" o "zona 20 (muy abajo)".

**Importancia:** Da una idea de cuán próximo está pájaro de la abertura entre caños, si un salto lo compromete y si hay tiempo suficiente de reacción.

### Ejemplo
