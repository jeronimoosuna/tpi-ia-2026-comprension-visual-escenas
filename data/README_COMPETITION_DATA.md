# TPI IA 2026 — estructura de los datos

Al crear una notebook desde la competencia, Kaggle monta esta fuente completa
de datos. No es necesario utilizar **Add Input** para incorporar el dataset de
desarrollo.

## Carpetas

- `development/`: 120 imágenes anotadas para EDA, experimentación, métricas
  locales y el benchmark de lenguaje.
- `tpi_ia_2026_competition_public/`: 60 imágenes de test sin anotaciones,
  vocabulario y metadatos propios de la competencia.
- `tpi_2026_qa_public/`: las 60 consignas del benchmark de lenguaje sobre 20
  imágenes de desarrollo. Sus respuestas se integran en el `SceneJSON` del
  envío; no se entrega un archivo QA separado.
- En la raíz: `scene_schema.json` y `sample_submission.csv`, porque el envío
  combina imágenes de test y del benchmark de desarrollo.

## Formato del envío

El `submission.csv` debe conservar exactamente las columnas `image_id` y
`SceneJSON` y tener 80 filas: una por cada imagen de test y una por cada una
de las 20 imágenes de desarrollo incluidas en `questions_dev.jsonl`.

En las 20 filas de desarrollo, `SceneJSON.questions` contiene las respuestas
del motor determinista y `SceneJSON.questions_llm` las respuestas del modo con
LLM. En las filas de test ambos campos pueden omitirse o ser listas vacías.

Las anotaciones privadas del test no forman parte de esta fuente y sólo son
accesibles para la métrica de Kaggle.

Dataset público original:

https://www.kaggle.com/datasets/martingra/ds-comprension-visual-de-escenas
