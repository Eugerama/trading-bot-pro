"""
Script de Demostración Rápida y Local.
Crea el archivo .env si no existe, verifica dependencias y ejecuta el sistema.
"""

import os
import sys
import shutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(PROJECT_DIR, ".env")
ENV_EXAMPLE = os.path.join(PROJECT_DIR, ".env.example")

# Crear .env si no existe
if not os.path.exists(ENV_FILE) and os.path.exists(ENV_EXAMPLE):
    print(">>> [Config] Creando archivo .env local a partir de .env.example...")
    shutil.copyfile(ENV_EXAMPLE, ENV_FILE)

# Verificación de dependencias clave
missing_packages = []
for pkg, import_name in [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("jinja2", "jinja2"),
    ("psutil", "psutil")
]:
    try:
        __import__(import_name)
    except ImportError:
        missing_packages.append(pkg)

if missing_packages:
    print("\n" + "=" * 65)
    print("⚠️ FALTAN DEPENDENCIAS CLAVE PARA INICIAR EL SERVIDOR")
    print(f"Paquetes faltantes: {', '.join(missing_packages)}")
    print("=" * 65)
    print("\nEjecuta el siguiente comando en tu terminal para instalar todo:\n")
    print("    pip install -r requirements.txt\n")
    print("O para instalar el entorno completo en modo aislado:")
    print("    python3 -m venv venv")
    print("    source venv/bin/activate")
    print("    pip install -r requirements.txt")
    print("=" * 65 + "\n")
    sys.exit(1)

print("Iniciando Quantitative Trading Bot & AI Dashboard...")
from main import main

if __name__ == "__main__":
    main()
