# 1. Crear entorno virtual
python3 -m venv venv

# 2. Activarlo
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate.bat   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar el torneo
python main.py

# 5. (Opcional) Visualizar partidas
python visualizar_partida.py
