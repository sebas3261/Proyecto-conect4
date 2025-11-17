# Connect 4 Tournament – IA Project

Este proyecto ejecuta un torneo automático entre diferentes políticas de Connect 4 y genera resultados en la carpeta `versus/`.

## 🚀 Instalación

### 1. Crear entorno virtual
```bash
python3 -m venv venv
```

### 2. Activar entorno virtual

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecutar el torneo

```bash
python main.py
```

Los resultados de cada partida se guardarán en la carpeta:

```
versus/
```

---

## 📊 (Opcional) Visualizar partidas

```bash
python visualizar_partida.py
```

---

## 📁 Estructura del proyecto
```
├── connect4/
├── groups/
├── tournament.py
├── main.py
├── visualizar_partida.py
├── requirements.txt
└── README.md
```

---

## ✔️ Requisitos

- Python 3.10+
- Entorno virtual (recomendado)

---

¡Listo! Copia este archivo como `README.md` en tu proyecto.
