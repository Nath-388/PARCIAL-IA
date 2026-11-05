# Calibración del umbral (coseno, SFace)

- Personas: 2 (dalyali: 5 fotos, nathalia: 4 fotos)
- Pares genuinos: 16 | pares impostores: 20
- Puntaje genuino: min 0.367, media 0.559, max 0.828
- Puntaje impostor: min 0.057, media 0.159, max 0.246

| Umbral | FAR | FRR | Impostores aceptados | Genuinos rechazados |
| ------ | --- | --- | -------------------- | ------------------- |
| 0.250 | 0.0% | 0.0% | 0/20 | 0/16 |
| 0.300 | 0.0% | 0.0% | 0/20 | 0/16 |
| 0.363 | 0.0% | 0.0% | 0/20 | 0/16 |
| 0.400 | 0.0% | 12.5% | 0/20 | 2/16 |
| 0.450 | 0.0% | 31.2% | 0/20 | 5/16 |

Menor umbral del barrido con FAR = 0% en estos datos: 0.250 (FRR = 0.0%).
Punto de cruce FAR≈FRR en el barrido: umbral 0.250 (FAR 0.0%, FRR 0.0%).

Elección del umbral: _(completar: valor elegido y por qué; en un login suele priorizarse un FAR bajo aceptando algo más de FRR, porque el usuario legítimo puede reintentar)_.
