# Parcial práctico. Control de gastos

**Asignatura:** Desarrollo de aplicaciones web
**Tema:** API de gastos con Express, MySQL, TypeScript, MVC y JWT, más un cliente mínimo
**Duración:** 1 hora y 30 minutos
**Valor total:** 100 puntos (5.0)

---

## Enunciado

Construya **desde cero** una aplicación web para registrar gastos personales. No parte de ningún laboratorio anterior: se crea la carpeta, el proyecto y todo el código durante el examen.

La aplicación tiene:

- Un servidor Express que entrega el documento HTML y responde una API JSON, ambos en el puerto **3000**.
- Los datos en MySQL.
- Un login que entrega un **JWT**. Sin token no se puede ver ni modificar ningún gasto.
- Un cliente con tres archivos: `js/model.js`, `js/view.js` y `js/app.js`.

### Reglas de arquitectura

- **Cliente:** `model.js` es el único archivo que llama a `fetch`, `view.js` es el único que dibuja y `app.js` conecta los dos.
- **Servidor:** el modelo, el controlador y la vista se escriben en **TypeScript** y se compilan con `tsc`. `server.js`, `db.js` y los archivos de rutas están en **JavaScript**.
- El SQL solo aparece en el modelo. El modelo no referencia `req` ni `res`. El controlador no contiene SQL. Las rutas solo asocian método y ruta con una función del controlador (no validan ni eligen status).
- Prohibido: `express-session`, cookies de sesión y guardar el token en `localStorage`, `sessionStorage` o cookies.

---

## Datos fijos

Use exactamente estos datos. No los cambie.

**Variables de `.env`** (cree este archivo, no se entrega):

```
PORT=3000
DB_HOST=localhost
DB_PORT=3306
DB_USER=su_usuario
DB_PASSWORD=su_clave
DB_NAME=parcial_gastos
JWT_SECRET=frase-secreta-del-parcial
```

**Script SQL** (guárdelo como `sql/schema.sql` y ejecútelo):

```sql
CREATE DATABASE IF NOT EXISTS parcial_gastos;
USE parcial_gastos;

CREATE TABLE IF NOT EXISTS users (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  email VARCHAR(255) NOT NULL,
  password VARCHAR(255) NOT NULL,
  UNIQUE KEY uq_users_email (email)
);

CREATE TABLE IF NOT EXISTS expenses (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  description VARCHAR(100) NOT NULL,
  amount DECIMAL(10,2) NOT NULL,
  category VARCHAR(20) NOT NULL
);

INSERT INTO users (email, password) VALUES ('luis@upb.edu.co', 'clave123');

INSERT INTO expenses (description, amount, category) VALUES
  ('Almuerzo', 18000, 'comida'),
  ('Bus', 2900, 'transporte'),
  ('Cuaderno', 7500, 'otros');
```

**Contrato HTTP:**

| Operación | Método + URL | Respuesta |
| --------- | ------------ | --------- |
| Login | `POST /login` con `{ email, password }` | **200** `{ token }` o **401** `{ error: 'Credenciales inválidas' }` |
| Listar | `GET /expenses` | **200**, arreglo `[{ id, description, amount, category }, ...]` |
| Crear | `POST /expenses` con `{ description, amount, category }` | **201**, objeto creado con `id`. **400** si es inválido |
| Eliminar | `DELETE /expenses/:id` | **200** sin cuerpo. **404** si no existe |
| Sin token válido | cualquiera bajo `/expenses` | **401** `{ error: 'No autenticado' }` |

**Validación del `POST /expenses`:**

- `description`: cadena no vacía tras `trim()`, máximo 100 caracteres.
- `amount`: número mayor que 0.
- `category`: uno de `'comida'`, `'transporte'`, `'otros'`.

Un error de base de datos responde **500** con `{ error: '...' }`.

**Token:** algoritmo `HS256`, secreto `JWT_SECRET`, expiración `20m`, payload con `sub` y `email` (nunca la contraseña).

---

## Parte 1. Proyecto y base de datos (10 puntos)

1. **(3 pts)** Cree la carpeta del proyecto, inicialice npm e instale `express`, `mysql2`, `dotenv`, `jsonwebtoken`, y como dependencias de desarrollo `typescript`, `@types/node`, `@types/express` y `@types/jsonwebtoken`. Dejé los scripts `build` (ejecuta `tsc`) y `api` (compila y arranca `server/server.js`).
2. **(3 pts)** Cree `tsconfig.json` con `strict`, módulo `commonjs`, código fuente en `server/src` y salida en `server/dist`.
3. **(2 pts)** Escriba `server/db.js` con un pool de `mysql2/promise` que lea las variables `DB_*`.
4. **(2 pts)** Ejecute `sql/schema.sql` y compruebe que `expenses` tiene 3 filas.

---

## Parte 2. Servidor MVC en TypeScript (35 puntos)

1. **(10 pts) Modelo** `server/src/models/expenseModel.ts`: `findAll()`, `findById(id)`, `create({ description, amount, category })` (devuelve el gasto creado con `id`) y `remove(id)` (devuelve `true` o `false`). Consultas parametrizadas con `?`.
2. **(15 pts) Controlador** `server/src/controllers/expenseController.ts`: `list`, `create` y `remove`. Valida según el enunciado, llama al modelo, elige el status y el JSON, responde **500** ante un fallo de base de datos y no contiene SQL.
3. **(5 pts) Vista** `server/src/views/expenseView.ts`: función que recibe una fila y devuelve el objeto JSON del contrato, con `amount` como **número** (MySQL devuelve los `DECIMAL` como texto).
4. **(5 pts) Rutas y arranque:** `server/routes/expenses.js` y `server/server.js` (JavaScript). `server.js` carga `dotenv`, usa `express.json()`, entrega la carpeta `public/` y escucha en `PORT`.

---

## Parte 3. Autenticación con JWT (25 puntos)

1. **(10 pts) Login:** modelo de usuarios (`userModel.ts`, con el único `SELECT` de `users`), controlador `authController.ts` y ruta `POST /login`. Debe responder **200** `{ token }` o **401**, firmar con los datos fijos y no devolver la contraseña.
2. **(10 pts) Middleware** `requireJwt`: lee `Authorization: Bearer <token>`, verifica con `jwt.verify` fijando `algorithms: ['HS256']` y responde **401** `{ error: 'No autenticado' }` si falta, tiene otro esquema, está mal firmado o venció.
3. **(5 pts) Protección:** monte `requireJwt` de modo que **todos** los métodos bajo `/expenses` queden protegidos y `POST /login` quede abierto.

---

## Parte 4. Cliente (20 puntos)

Cree `public/index.html`, `public/js/model.js`, `public/js/view.js` y `public/js/app.js` (el diseño es libre, pero el HTML debe tener un `#status`, un formulario de login, un formulario para agregar gastos, la lista y el total).

1. **(8 pts) `model.js`:** variable de módulo `token`; `login(email, password)` (guarda el token y lanza error si `response.ok` es falso); `getExpenses()`, `createExpense(...)` y `deleteExpense(id)`, todas con `Authorization: Bearer`; `logout()` que borra el token de la variable. No usa `document`.
2. **(6 pts) `view.js`:** `showLogin()`, `showApp()` y `renderExpenses(expenses)`, que dibuja cada gasto con `createElement` y `textContent` (con botón Eliminar) y escribe el **total** (suma de `amount`). Nada de `fetch`.
3. **(6 pts) `app.js`:** al cargar, solo se ve el login. Tras un login exitoso se muestran los gastos. Agregar y eliminar vuelven a pedir `GET /expenses` para dibujar. Los errores se escriben en `#status`.

---

## Parte 5. Preguntas (10 puntos)

Responda cada una en máximo 3 líneas (2 puntos cada una).

1. ¿Por qué el login se hace con `POST` y no con `GET`?
2. Si el cliente llamara al servidor desde otro puerto (por ejemplo, `:5500`), ¿qué problema aparecería y por qué no aparece en este examen?
3. ¿Qué sucede con el token al recargar la página? ¿Por qué?
4. ¿Por qué el controlador no debe contener SQL?
5. Dos usuarios usan la misma cuenta. Si uno hace `POST /logout`, ¿cambia la validez de su token? Explique cómo funciona un JWT en ese caso.

---

## Verificación en clase

- [ ] `npm run api` arranca sin errores y `http://localhost:3000/` muestra el login (Go Live apagado).
- [ ] Sin token, `GET`, `POST` y `DELETE` bajo `/expenses` responden **401**.
- [ ] `POST /login` con `luis@upb.edu.co` / `clave123` responde **200** `{ token }` sin contraseña. Con otra clave responde **401**.
- [ ] Tras el login se ven los 3 gastos y el total `28400`.
- [ ] Crear un gasto válido responde **201** y persiste en MySQL. Con `amount: -5` o `category: 'viajes'` responde **400**.
- [ ] Eliminar un gasto responde **200**. Un `id` inexistente responde **404**.
- [ ] En **Network**, `/expenses` lleva `Authorization: Bearer`. En **Application** no hay cookie de sesión ni token en `localStorage`.
- [ ] Tras recargar, se ve el login y `GET /expenses` responde **401**.
- [ ] El SQL está solo en los modelos. El controlador no tiene SQL. Las rutas no validan.
- [ ] `package.json` no depende de `express-session`.

---

## Rúbrica

| Parte | Puntos |
| ----- | ------ |
| 1. Proyecto y base de datos | 10 |
| 2. Servidor MVC en TypeScript | 35 |
| 3. JWT | 25 |
| 4. Cliente | 20 |
| 5. Preguntas | 10 |
| **Total** | **100** |

**Penalizaciones:** SQL fuera del modelo (-5), validación en las rutas (-3), token en almacenamiento del navegador (-5), contraseña en el payload o en la respuesta (-5), `express-session` (-5).

**Sugerencia de tiempo:** Parte 1 (10 min), Parte 2 (30 min), Parte 3 (20 min), Parte 4 (20 min), Parte 5 (5 min), repaso (5 min).

---

## Forma de entrega

Entregar dos archivos (no unirlos):

- `parcial-NOMBRE-ESTUDIANTE.md` con el código del servidor, el código del cliente y las respuestas de la Parte 5. No va dentro del `.zip`.
- `parcial-NOMBRE-ESTUDIANTE.zip` con el proyecto: `package.json`, `tsconfig.json`, `.env.example`, `sql/schema.sql`, `public/` (HTML y `js/`), `server/server.js`, `server/db.js`, `server/routes/` y `server/src/` (TypeScript).

No incluir `.env`, `node_modules/` ni `server/dist/`.
