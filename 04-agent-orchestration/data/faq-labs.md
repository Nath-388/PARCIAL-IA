# Preguntas frecuentes sobre los laboratorios

## ¿Qué necesito instalar para los laboratorios?

Se necesita Python 3.11 o superior y uv, el gestor de entornos y dependencias de Python. Desde la
raíz del repositorio se ejecuta uv sync para instalar las dependencias. Si no se puede instalar
nada en el equipo, se puede trabajar en GitHub Codespaces.

## El programa dice "Falta LLM_API_KEY"

El archivo .env no existe o no tiene la clave. Hay que copiar .env.example como .env y escribir la
API key del proveedor en la variable LLM_API_KEY. El archivo .env debe estar en la raíz del
repositorio.

## Aparece un error 401 del proveedor

Un error 401 significa que el proveedor rechazó la API key: la clave está mal copiada, fue revocada
o corresponde a otro proveedor distinto al configurado en LLM_PROVIDER. Hay que revisar que la
clave y el proveedor coincidan.

## Aparece un error 429 del proveedor

Un error 429 significa que se superó el límite de peticiones o de tokens del plan gratuito. Hay que
esperar un minuto y volver a intentar, o cambiar a un modelo más pequeño en LLM_MODEL.

## ¿Puedo usar otro proveedor u otro modelo?

Sí. Los laboratorios funcionan con Groq, OpenRouter, OpenAI y DeepSeek. Para cambiar de proveedor
solo se modifican LLM_PROVIDER, LLM_API_KEY y LLM_MODEL en el archivo .env, sin cambiar el código.

## La primera ejecución del laboratorio 2 es lenta

La primera vez que se calculan embeddings se descarga el modelo de embeddings, de unos 220 MB, y
se guarda en caché. Las ejecuciones siguientes usan la copia local y no requieren descargarlo de
nuevo.

## ¿Dónde se entregan los laboratorios?

Cada laboratorio se entrega en la tarea de Microsoft Teams correspondiente, antes de la fecha
límite indicada en la tarea. No se reciben entregas por correo electrónico.
