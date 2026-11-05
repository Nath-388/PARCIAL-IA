# Parcial práctico. Corte 2

**Asignatura:** Desarrollo de aplicaciones web
**Tema:** API de reseñas del catálogo UPB Bookstore (TypeScript, MVC en el servidor, HTTP, MySQL y JWT)
**Duración sugerida:** 2 horas 30 minutos
**Valor total:** 100 puntos (5.0)

---

## Instrucciones generales

- Parta del proyecto terminado del Lab 18 (catálogo con JWT, `npm run api`, MySQL con las tablas `books` y `users`).
- El examen agrega al catálogo un nuevo recurso: **reseñas** (`reviews`) de los libros. Solo puede usarlas quien haya iniciado sesión.
- Se conserva la arquitectura de los laboratorios:
  - Cliente: `js/model.js` (único con `fetch`), `js/view.js` (dibuja) y `js/app.js` (controlador).
  - Servidor: modelo, controlador y vista en **TypeScript** compilados con `tsc`. `server.js` y los archivos de rutas siguen en **JavaScript**.
- No se permite `express-session`, ni guardar el token en `localStorage`, `sessionStorage` o cookies.
- El SQL solo puede aparecer en el modelo del servidor. El modelo no referencia `req` ni `res`. Las rutas no validan el cuerpo ni eligen el status.
- Sin Go Live: el documento se abre desde `http://localhost:3000/`.
- Todo texto de respuesta de las preguntas conceptuales debe ser breve (máximo 4 líneas por pregunta).

---

## Datos fijos

Tabla de reseñas:

```sql
CREATE TABLE reviews (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  book_id INT UNSIGNED NOT NULL,
  rating TINYINT UNSIGNED NOT NULL,
  comment VARCHAR(255) NOT NULL,
  CONSTRAINT fk_reviews_book FOREIGN KEY (book_id) REFERENCES books(id)
);
```

Datos iniciales (`seed`):

| id | book_id | rating | comment                  |
| -- | ------- | ------ | ------------------------ |
| 1  | 1       | 5      | Lectura obligatoria      |
| 2  | 1       | 4      | Buenos ejemplos          |
| 3  | 4       | 3      | Corto pero útil          |

Contrato HTTP del recurso (todos bajo `/reviews` y todos exigen `Authorization: Bearer`):

| Operación       | Método + URL          | Respuesta                                                        |
| --------------- | --------------------- | ---------------------------------------------------------------- |
| Listar          | `GET /reviews`        | **200**, arreglo `[{ id, book_id, rating, comment }, ...]`       |
| Crear           | `POST /reviews`       | **201**, objeto creado con `id`. **400** si el cuerpo no es válido |
| Cambiar puntaje | `PATCH /reviews/:id`  | **200**, objeto actualizado. **400** o **404**                   |
| Eliminar        | `DELETE /reviews/:id` | **200** sin cuerpo. **404** si no existe                         |
| Sin token válido| cualquiera            | **401** `{ error: 'No autenticado' }`                            |

Reglas de validación del `POST`: `book_id` entero positivo, `rating` entero entre 1 y 5, `comment` cadena no vacía tras `trim()`.

Token JWT: algoritmo `HS256`, secreto `JWT_SECRET`, expiración `30m`, payload con `sub` y `email` (nunca la contraseña).

---

## Parte 1. TypeScript (15 puntos)

Cree el archivo independiente `src/parcial-ts.ts` (no importa otros archivos) y compílelo con `strict`.

**1.1 (4 pts)** Declare una `interface Review` con los campos `id: number`, `bookId: number`, `rating: number`, `comment: string` y un campo **opcional** `author`.

**1.2 (4 pts)** Con la palabra `function`, escriba `promedio(reseñas: Review[]): number` que devuelva el promedio de `rating` (0 si el arreglo está vacío). Declare tipos explícitos en parámetros y retorno.

**1.3 (4 pts)** Escriba la clase `ReseñaRegistrada` que **implemente** `Review`. Debe tener: un campo `private` llamado `votos`, un campo `protected` llamado `origen`, y un método `votar(): void` (sin la palabra `function`) que sume 1 a `votos`.

**1.4 (3 pts)** Defina un tipo `Estado` como unión de los literales `'visible' | 'oculta'` y escriba `etiqueta(estado: Estado): string` que devuelva `'Se muestra'` o `'No se muestra'` usando un solo `if` y un `return` final, sin `else`.

---

## Parte 2. HTTP (10 puntos)

**2.1 (4 pts)** Complete la tabla (una frase breve por fila) con el significado de cada código de estado y diga en qué caso del contrato de `/reviews` se usa.

| Código | Significado | Caso en `/reviews` |
| ------ | ----------- | ------------------ |
| 200    |             |                    |
| 201    |             |                    |
| 400    |             |                    |
| 401    |             |                    |
| 404    |             |                    |
| 500    |             |                    |

**2.2 (3 pts)** Una con una flecha cada operación CRUD con su método HTTP: Create, Read, Update, Delete. Explique en una línea por qué se usa `PATCH` y no `PUT` para cambiar solo el puntaje.

**2.3 (3 pts)** `POST /login` devuelve **200** o **401**. Explique por qué el login se hace con `POST` y no con `GET`.

---

## Parte 3. Servidor MVC en TypeScript (30 puntos)

**3.1 (5 pts) Mapa de roles.** Proponga la estructura de carpetas y archivos del servidor para `/reviews` (modelo, controlador, vista, archivo de rutas). Indique cuáles son `.ts` y cuáles `.js`.

**3.2 (10 pts) Modelo.** En `reviewModel.ts` implemente: `findAll()`, `findById(id)`, `create({ book_id, rating, comment })`, `updateRating(id, rating)` (devuelve `null` si no existe) y `remove(id)` (devuelve `true` o `false`). Debe usar consultas parametrizadas con `?`, y no puede referenciar `req`, `res` ni HTML.

**3.3 (10 pts) Controlador.** En `reviewController.ts` implemente `list`, `create`, `updateRating` y `remove`. Debe validar la petición según las reglas del enunciado, llamar al modelo, elegir el status y el JSON, responder **500** `{ error: '...' }` ante un fallo de base de datos y no contener SQL.

**3.4 (3 pts) Vista.** En `reviewView.ts` escriba una función que reciba una fila de la base de datos y devuelva el objeto JSON que se enviará (`id`, `book_id`, `rating`, `comment`).

**3.5 (2 pts) Rutas.** Escriba `routes/reviews.js` (JavaScript) que solo asocie cada método y ruta con una función del controlador. No valida el cuerpo ni elige el status.

---

## Parte 4. Autenticación con JWT (25 puntos)

**4.1 (8 pts) Login.** Escriba el controlador de `POST /login` (TypeScript) que:

- valide que `email` y `password` lleguen como cadenas;
- consulte la tabla `users` **a través del modelo** (el controlador no tiene SQL);
- responda **401** `{ error: 'Credenciales inválidas' }` si no coincide;
- si coincide, firme un JWT con `sub` y `email`, `HS256`, `30m` y `JWT_SECRET`, y responda **200** `{ token }` sin contraseña.

**4.2 (8 pts) Middleware.** Escriba `requireJwt` que lea `Authorization: Bearer <token>`, verifique con `jwt.verify` forzando `algorithms: ['HS256']` y responda **401** `{ error: 'No autenticado' }` ante ausencia, esquema incorrecto, firma inválida o token vencido. Si es válido, llama a `next()`.

**4.3 (4 pts) Protección.** Muestre cómo se monta `requireJwt` para que **todos** los métodos bajo `/reviews` (y `/books`) queden protegidos, mientras `POST /login` y `POST /logout` siguen abiertos.

**4.4 (5 pts) Logout.** Escriba `POST /logout` y responda: tras `POST /logout`, ¿el token ya entregado sigue siendo válido? ¿Por qué? ¿Qué debería hacer el cliente para "cerrar sesión"?

---

## Parte 5. Cliente MVC (15 puntos)

**5.1 (6 pts) `js/model.js`.** Implemente con `fetch`:

- una variable de módulo `token` (inicia en `null`);
- `login(email, password)` que haga `POST /login`, guarde el token en esa variable y lance un error si `response.ok` es falso;
- `getReviews()` y `createReview({ book_id, rating, comment })` que envíen `Authorization: Bearer` en cada solicitud;
- `logout()` que haga `POST /logout` y ponga `token = null`.

**5.2 (5 pts) `js/view.js` y `js/app.js`.** Describa (con código breve) cómo:

- sin token, solo se muestra el formulario de login y se oculta la zona de reseñas;
- tras un `POST /login` exitoso se oculta el login y se dibujan las reseñas con `createElement` y `textContent`;
- `app.js` captura el error de login y escribe en `#status`.

**5.3 (4 pts)** Después de crear una reseña con éxito, ¿por qué `app.js` vuelve a llamar a `getReviews()` en lugar de hacer `push` al arreglo en memoria?

---

## Parte 6. Preguntas de síntesis (5 puntos)

Responda cada una en máximo 4 líneas (1 punto cada una).

1. Si el documento HTML se sirve con Live Server (`:5500`) y la API corre en `:3000`, ¿qué problema aparece y qué mecanismo lo resuelve? ¿Por qué deja de ser necesario cuando Express entrega el documento?
2. ¿Por qué el token no debe guardarse en `localStorage`? ¿Qué sucede con él al recargar la página en esta arquitectura?
3. ¿Qué diferencia hay entre una sesión con `express-session` y un JWT en cuanto a **dónde vive el estado**?
4. ¿Por qué el payload del JWT no debe contener la contraseña?
5. ¿Qué archivo del servidor es el único que puede conocer el texto de un `SELECT`?

---

## Lista de verificación en clase

- [ ] `npm run build` (o `tsc`) compila sin errores los archivos del servidor y `src/parcial-ts.ts`.
- [ ] `GET`, `POST`, `PATCH` y `DELETE` bajo `/reviews` responden **401** sin token.
- [ ] `POST /login` responde **200** con `{ token }` y sin contraseña.
- [ ] Con token válido, se listan, crean, modifican y eliminan reseñas, y los cambios persisten en MySQL.
- [ ] Un `POST /reviews` con `rating: 9` o `comment` vacío responde **400**.
- [ ] Un `PATCH` o `DELETE` sobre un `id` inexistente responde **404**.
- [ ] En **Network**, `/reviews` lleva `Authorization: Bearer` y **Application** no muestra cookie de sesión ni `localStorage` con el token.
- [ ] Tras `POST /logout`, el token copiado sigue obteniendo **200**.
- [ ] Tras recargar, se ve el formulario de login y `GET /reviews` responde **401**.
- [ ] El SQL está solo en el modelo. El controlador no tiene SQL. Las rutas no validan ni eligen status.
- [ ] `package.json` no depende de `express-session`.

---

## Rúbrica resumida

| Parte | Puntos |
| ----- | ------ |
| 1. TypeScript | 15 |
| 2. HTTP | 10 |
| 3. Servidor MVC en TypeScript | 30 |
| 4. JWT | 25 |
| 5. Cliente MVC | 15 |
| 6. Síntesis | 5 |
| **Total** | **100** |

**Penalizaciones:** SQL fuera del modelo (-5), validación en las rutas (-3), token guardado en almacenamiento del navegador (-5), contraseña en el payload o en la respuesta (-5), uso de `express-session` (-5).

---

## Forma de entrega

Entregar dos archivos (no unirlos):

- `parcial-solucion-NOMBRE-ESTUDIANTE.md` con las respuestas escritas, el mapa de roles y todo el código solicitado. No va dentro del `.zip`.
- `parcial-solucion-NOMBRE-ESTUDIANTE.zip` con el proyecto que arranca con `npm run api`: documento HTML, `css/`, `js/`, `package.json`, `.env.example`, `tsconfig.json`, scripts SQL (`books`, `users`, `reviews`), `server.js` y rutas en JavaScript, y modelo, controladores y vista en TypeScript, más `src/parcial-ts.ts`.

No incluir `.env`, `node_modules/` ni `dist/`.
