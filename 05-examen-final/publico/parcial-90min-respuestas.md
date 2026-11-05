# Solucionario. Parcial práctico: Control de gastos (90 min)

**Uso exclusivo del docente.** El código es una referencia: se acepta cualquier implementación equivalente que cumpla el contrato y el reparto MVC.

---

## Estructura de referencia

```
parcial-gastos/
|--- package.json
|--- tsconfig.json
|--- .env.example
|--- sql/schema.sql
|--- public/
|    |--- index.html
|    |--- js/model.js
|    |--- js/view.js
|    |--- js/app.js
|--- server/
     |--- server.js
     |--- db.js
     |--- routes/expenses.js
     |--- routes/auth.js
     |--- src/
          |--- models/expenseModel.ts
          |--- models/userModel.ts
          |--- controllers/expenseController.ts
          |--- controllers/authController.ts
          |--- views/expenseView.ts
          |--- middleware/requireJwt.ts
```

---

## Parte 1. Proyecto y base de datos (10 pts)

**1. (3 pts)**

```
npm init -y
npm install express mysql2 dotenv jsonwebtoken
npm install -D typescript @types/node @types/express @types/jsonwebtoken
npx tsc --init
npm run api
```

`package.json` (scripts):

```json
"scripts": {
  "build": "tsc",
  "api": "tsc && node server/server.js"
}
```

Criterios: dependencias correctas (2 pts), scripts `build` y `api` (1 pt).

**2. (3 pts)** `tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "commonjs",
    "strict": true,
    "esModuleInterop": true,
    "rootDir": "server/src",
    "outDir": "server/dist",
    "noEmitOnError": true
  },
  "include": ["server/src"]
}
```

**3. (2 pts)** `server/db.js`:

```javascript
//server/db.js
require('dotenv').config();
const mysql = require('mysql2/promise');

const pool = mysql.createPool({
  host: process.env.DB_HOST,
  port: Number(process.env.DB_PORT),
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
});

module.exports = pool;
```

**4. (2 pts)** `mysql -u USUARIO -p < sql/schema.sql` y `SELECT COUNT(*) FROM expenses;` devuelve 3.

---

## Parte 2. Servidor MVC en TypeScript (35 pts)

### Modelo (10 pts)

```typescript
//server/src/models/expenseModel.ts
const pool = require('../../db');

export interface ExpenseRow {
  id: number;
  description: string;
  amount: string | number;   //DECIMAL llega como texto
  category: string;
}

export async function findAll(): Promise<ExpenseRow[]> {
  const [rows] = await pool.query(
    'SELECT id, description, amount, category FROM expenses ORDER BY id'
  );
  return rows as ExpenseRow[];
}

export async function findById(id: number): Promise<ExpenseRow | null> {
  const [rows] = await pool.query(
    'SELECT id, description, amount, category FROM expenses WHERE id = ?',
    [id]
  );
  const list = rows as ExpenseRow[];
  return list.length > 0 ? list[0] : null;
}

export async function create(data: {
  description: string;
  amount: number;
  category: string;
}): Promise<ExpenseRow | null> {
  const [result] = await pool.query(
    'INSERT INTO expenses (description, amount, category) VALUES (?, ?, ?)',
    [data.description, data.amount, data.category]
  );
  return findById((result as { insertId: number }).insertId);
}

export async function remove(id: number): Promise<boolean> {
  const [result] = await pool.query('DELETE FROM expenses WHERE id = ?', [id]);
  return (result as { affectedRows: number }).affectedRows > 0;
}
```

Criterios: cuatro funciones (4 pts), `?` en todas las consultas (4 pts), sin `req`, `res` ni HTML (2 pts).

### Controlador (15 pts)

```typescript
//server/src/controllers/expenseController.ts
import { Request, Response } from 'express';
import * as model from '../models/expenseModel';
import { toJson } from '../views/expenseView';

const CATEGORIES = ['comida', 'transporte', 'otros'];

export async function list(req: Request, res: Response): Promise<void> {
  try {
    const rows = await model.findAll();
    res.status(200).json(rows.map(toJson));
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error al leer los gastos' });
  }
}

export async function create(req: Request, res: Response): Promise<void> {
  const { description, amount, category } = req.body ?? {};
  const cleanDescription = typeof description === 'string' ? description.trim() : '';
  const validDescription = cleanDescription !== '' && cleanDescription.length <= 100;
  const validAmount = typeof amount === 'number' && Number.isFinite(amount) && amount > 0;
  const validCategory = CATEGORIES.includes(category);
  if (!validDescription || !validAmount || !validCategory) {
    res.status(400).json({ error: 'Datos inválidos' });
    return;
  }
  try {
    const row = await model.create({
      description: cleanDescription,
      amount,
      category,
    });
    res.status(201).json(toJson(row!));
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error al guardar el gasto' });
  }
}

export async function remove(req: Request, res: Response): Promise<void> {
  const id = Number(req.params.id);
  if (!Number.isInteger(id) || id <= 0) {
    res.status(404).json({ error: 'No existe' });
    return;
  }
  try {
    const removed = await model.remove(id);
    if (!removed) {
      res.status(404).json({ error: 'No existe' });
      return;
    }
    res.sendStatus(200);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error al eliminar el gasto' });
  }
}
```

Criterios: validación de los tres campos (5 pts), status 200/201/400/404 correctos (5 pts), 500 con JSON breve y `console.error` (3 pts), sin SQL (2 pts).

### Vista (5 pts)

```typescript
//server/src/views/expenseView.ts
import { ExpenseRow } from '../models/expenseModel';

export function toJson(row: ExpenseRow): {
  id: number;
  description: string;
  amount: number;
  category: string;
} {
  return {
    id: row.id,
    description: row.description,
    amount: Number(row.amount),
    category: row.category,
  };
}
```

Criterios: devuelve solo los campos del contrato (2 pts), convierte `amount` a número (2 pts), sin HTML (1 pt).

### Rutas y arranque (5 pts)

```javascript
//server/routes/expenses.js
const express = require('express');
const controller = require('../dist/controllers/expenseController');

const router = express.Router();

router.get('/', controller.list);
router.post('/', controller.create);
router.delete('/:id', controller.remove);

module.exports = router;
```

```javascript
//server/server.js
require('dotenv').config();
const path = require('path');
const express = require('express');
const { requireJwt } = require('./dist/middleware/requireJwt');

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, '..', 'public')));

app.use('/', require('./routes/auth'));                               //login: abierto
app.use('/expenses', requireJwt, require('./routes/expenses'));       //protegido

const port = Number(process.env.PORT) || 3000;
app.listen(port, () => {
  console.log('Escuchando en el puerto ' + port);
});
```

Criterios: rutas solo asocian método y ruta (2 pts), `express.json()` y `express.static` (2 pts), escucha en `PORT` (1 pt).

---

## Parte 3. JWT (25 pts)

### Login (10 pts)

```typescript
//server/src/models/userModel.ts
const pool = require('../../db');

export interface UserRow {
  id: number;
  email: string;
  password: string;
}

export async function findByCredentials(
  email: string,
  password: string
): Promise<UserRow | null> {
  const [rows] = await pool.query(
    'SELECT id, email, password FROM users WHERE email = ? AND password = ?',
    [email, password]
  );
  const list = rows as UserRow[];
  return list.length > 0 ? list[0] : null;
}
```

```typescript
//server/src/controllers/authController.ts
import { Request, Response } from 'express';
import jwt from 'jsonwebtoken';
import * as users from '../models/userModel';

export async function login(req: Request, res: Response): Promise<void> {
  const { email, password } = req.body ?? {};
  if (typeof email !== 'string' || typeof password !== 'string') {
    res.status(401).json({ error: 'Credenciales inválidas' });
    return;
  }
  try {
    const user = await users.findByCredentials(email, password);
    if (user === null) {
      res.status(401).json({ error: 'Credenciales inválidas' });
      return;
    }
    const token = jwt.sign(
      { sub: String(user.id), email: user.email },
      process.env.JWT_SECRET as string,
      { expiresIn: '20m', algorithm: 'HS256' }
    );
    res.status(200).json({ token });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error de servidor' });
  }
}
```

```javascript
//server/routes/auth.js
const express = require('express');
const controller = require('../dist/controllers/authController');

const router = express.Router();
router.post('/login', controller.login);

module.exports = router;
```

Criterios: `SELECT` solo en el modelo (3 pts), 401 correcto (2 pts), `jwt.sign` con `sub`, `email`, `HS256`, `20m` y `JWT_SECRET` (4 pts), respuesta `{ token }` sin contraseña (1 pt).

### Middleware (10 pts)

```typescript
//server/src/middleware/requireJwt.ts
import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';

export function requireJwt(req: Request, res: Response, next: NextFunction): void {
  const header = req.headers.authorization || '';
  const [scheme, token] = header.split(' ');
  if (scheme !== 'Bearer' || !token) {
    res.status(401).json({ error: 'No autenticado' });
    return;
  }
  try {
    jwt.verify(token, process.env.JWT_SECRET as string, { algorithms: ['HS256'] });
    next();
  } catch {
    res.status(401).json({ error: 'No autenticado' });
  }
}
```

Criterios: separa el encabezado y valida `Bearer` (3 pts), `jwt.verify` con `algorithms: ['HS256']` (4 pts), 401 con el JSON pedido en todos los fallos (2 pts), `next()` si es válido (1 pt).

### Protección (5 pts)

Está en `server.js`: `app.use('/expenses', requireJwt, ...)` se monta antes del router de gastos (4 pts) y `/login` queda fuera de la protección (1 pt). Alternativa válida: `router.use(requireJwt)` dentro de `routes/expenses.js`.

---

## Parte 4. Cliente (20 pts)

### `public/index.html` (referencia)

```html
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>Control de gastos</title>
</head>
<body>
  <h1>Control de gastos</h1>
  <p id="status"></p>

  <section id="login">
    <form id="login-form">
      <input id="email" type="email" placeholder="Correo" required />
      <input id="password" type="password" placeholder="Clave" required />
      <button type="submit">Entrar</button>
    </form>
  </section>

  <section id="app" hidden>
    <form id="expense-form">
      <input id="description" placeholder="Descripción" required />
      <input id="amount" type="number" step="0.01" min="0" placeholder="Monto" required />
      <select id="category">
        <option value="comida">comida</option>
        <option value="transporte">transporte</option>
        <option value="otros">otros</option>
      </select>
      <button type="submit">Agregar</button>
    </form>
    <ul id="expenses"></ul>
    <p>Total: <span id="total">0</span></p>
    <button type="button" id="logout">Salir</button>
  </section>

  <script type="module" src="js/app.js"></script>
</body>
</html>
```

### `model.js` (8 pts)

```javascript
//public/js/model.js
let token = null;

function authHeaders(extra) {
  return { ...extra, Authorization: 'Bearer ' + token };
}

export async function login(email, password) {
  const response = await fetch('/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  if (!response.ok) {
    throw new Error('HTTP ' + response.status);
  }
  const data = await response.json();
  token = data.token;
}

export async function getExpenses() {
  const response = await fetch('/expenses', { headers: authHeaders({}) });
  if (!response.ok) {
    throw new Error('HTTP ' + response.status);
  }
  return await response.json();
}

export async function createExpense({ description, amount, category }) {
  const response = await fetch('/expenses', {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ description, amount, category }),
  });
  if (!response.ok) {
    throw new Error('HTTP ' + response.status);
  }
  return await response.json();
}

export async function deleteExpense(id) {
  const response = await fetch('/expenses/' + id, {
    method: 'DELETE',
    headers: authHeaders({}),
  });
  if (!response.ok) {
    throw new Error('HTTP ' + response.status);
  }
}

export function logout() {
  token = null;
}
```

Criterios: token en variable de módulo (1 pt), `login` guarda el token y lanza error si falla (2 pts), `Authorization: Bearer` en las tres funciones de datos (3 pts), `Content-Type` y `JSON.stringify` en el `POST` (1 pt), `logout` limpia la variable (1 pt). Sin `document` (condición de validez).

### `view.js` (6 pts)

```javascript
//public/js/view.js
export function showLogin() {
  document.getElementById('login').hidden = false;
  document.getElementById('app').hidden = true;
}

export function showApp() {
  document.getElementById('login').hidden = true;
  document.getElementById('app').hidden = false;
}

export function renderExpenses(expenses) {
  const list = document.getElementById('expenses');
  list.replaceChildren();
  let total = 0;
  for (const e of expenses) {
    total = total + e.amount;
    const li = document.createElement('li');
    li.textContent = e.description + ' (' + e.category + '): ' + e.amount + ' ';
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'delete';
    button.dataset.id = e.id;
    button.textContent = 'Eliminar';
    li.appendChild(button);
    list.appendChild(li);
  }
  document.getElementById('total').textContent = String(total);
}
```

Criterios: `showLogin` y `showApp` (2 pts), dibujo con `createElement` y `textContent` (2 pts), total correcto (28400 con los datos iniciales) (2 pts). Sin `fetch`.

### `app.js` (6 pts)

```javascript
//public/js/app.js
import { login, getExpenses, createExpense, deleteExpense, logout } from './model.js';
import { showLogin, showApp, renderExpenses } from './view.js';

const statusEl = document.getElementById('status');

async function loadExpenses() {
  renderExpenses(await getExpenses());
}

showLogin();   //sin token: solo se ve el login

document.getElementById('login-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    await login(
      document.getElementById('email').value.trim(),
      document.getElementById('password').value
    );
    statusEl.textContent = '';
    showApp();
    await loadExpenses();
  } catch {
    statusEl.textContent = 'No se pudo iniciar sesión.';
  }
});

document.getElementById('expense-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    await createExpense({
      description: document.getElementById('description').value.trim(),
      amount: Number(document.getElementById('amount').value),
      category: document.getElementById('category').value,
    });
    event.target.reset();
    statusEl.textContent = '';
    await loadExpenses();
  } catch {
    statusEl.textContent = 'No se pudo guardar el gasto.';
  }
});

document.getElementById('expenses').addEventListener('click', async (event) => {
  const button = event.target.closest('.delete');
  if (!button) {
    return;
  }
  try {
    await deleteExpense(button.dataset.id);
    statusEl.textContent = '';
    await loadExpenses();
  } catch {
    statusEl.textContent = 'No se pudo eliminar el gasto.';
  }
});

document.getElementById('logout').addEventListener('click', () => {
  logout();
  showLogin();
});
```

Criterios: login visible al cargar y zona de gastos oculta (2 pts), agregar y eliminar vuelven a pedir `GET /expenses` (2 pts), errores en `#status` (2 pts).

---

## Parte 5. Preguntas (10 pts, 2 c/u)

1. **`POST` y no `GET`.** `GET` coloca los datos en la URL, donde quedan en el historial, los registros del servidor y los encabezados, lo que expondría la contraseña. `POST` las envía en el cuerpo. Además el login genera algo nuevo (un token) y no es una simple lectura.
2. **Otro puerto.** Es otro origen (el puerto forma parte del origen), así que el navegador aplica la política de mismo origen y bloquea la respuesta salvo que el servidor habilite CORS. En el examen el documento y la API los entrega Express en el puerto 3000, por lo que comparten origen y CORS no hace falta.
3. **Recarga.** El token se pierde, porque vive en una variable del módulo `model.js` (memoria de la página) y no en almacenamiento persistente. Al recargar se vuelve a ver el login y `GET /expenses` responde **401**.
4. **Sin SQL en el controlador.** Porque cada capa tiene una sola responsabilidad: el modelo conoce la base de datos, el controlador valida y decide el status. Si el SQL está solo en el modelo, cambiar de motor o de consulta no obliga a tocar el controlador y cada capa se puede probar por separado.
5. **Logout.** No cambia: el servidor no guarda el token, solo lo firmó, así que `POST /logout` no puede invalidarlo. El token sigue válido hasta que vence (20 minutos) para cualquiera que conserve la cadena. Cerrar sesión consiste en que el cliente descarte el token.

---

## Lista de verificación para el docente

| Ítem | Comprobación |
| ---- | ------------ |
| Arranque | `npm run api` compila y escucha en el 3000 |
| 401 sin token | `GET`, `POST` y `DELETE` sobre `/expenses` sin encabezado |
| Login | `luis@upb.edu.co` / `clave123` da **200** `{ token }`. Clave errónea da **401** |
| Datos | Tras el login se ven 3 gastos y el total `28400` |
| Alta | Válida da **201** y persiste. `amount: -5` o `category: 'viajes'` dan **400** |
| Baja | `DELETE` válido da **200**, un `id` inexistente da **404** |
| Cabeceras | En **Network**, `/expenses` lleva `Authorization: Bearer` |
| Almacenamiento | Sin cookie de sesión y sin token en `localStorage` |
| Recarga | Aparece el login y `GET /expenses` da **401** |
| Capas | `SELECT`, `INSERT` y `DELETE FROM` solo aparecen en los modelos |
| Dependencias | `package.json` sin `express-session` |

## Errores frecuentes y descuento

| Error | Descuento |
| ----- | --------- |
| SQL en el controlador o en las rutas | -5 |
| Validación o elección del status en las rutas | -3 |
| Token en `localStorage`, `sessionStorage` o cookie | -5 |
| `jwt.verify` sin fijar `algorithms` | -2 |
| Contraseña en el payload o en la respuesta del login | -5 |
| `express-session` en `package.json` | -5 |
| Concatenar valores en el SQL en lugar de usar `?` | -3 |
| `amount` enviado como texto por la vista del servidor | -2 |
| `model.js` usando `document` | -3 |
| Afirmar que `POST /logout` invalida el token | -2 |
