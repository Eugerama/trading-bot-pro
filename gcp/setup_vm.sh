#!/bin/bash
# ==============================================================================
# SCRIPT DE APROVISIONAMIENTO AUTOMÁTICO EN GOOGLE CLOUD COMPUTE ENGINE
# Sistema Operativo Objetivo: Ubuntu 22.04 LTS / Debian 12 (Serie E2: e2-medium)
# ==============================================================================

set -e

echo ">>> [1/5] Actualizando repositorios del sistema en la VM de GCP..."
sudo apt-get update -y && sudo apt-get upgrade -y

echo ">>> [2/5] Instalando dependencias base (Python 3, pip, tmux, git, ufw)..."
sudo apt-get install -y python3 python3-pip python3-venv git tmux screen curl htop ufw

echo ">>> [3/5] Configurando el entorno virtual de Python en /opt/trading-bot-pro..."
sudo mkdir -p /opt/trading-bot-pro
sudo chown -R $USER:$USER /opt/trading-bot-pro

# Copiar archivos del proyecto al directorio de producción
cp -r . /opt/trading-bot-pro/
cd /opt/trading-bot-pro

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ">>> [4/5] Configurando Firewall de GCP y local para el puerto Web (8080)..."
sudo ufw allow 8080/tcp || true
sudo ufw allow 22/tcp || true

echo ">>> [5/5] Instalando servicio de ejecución continua systemd (24/7 persistente)..."
sudo cp gcp/tradingbot.service /etc/systemd/system/tradingbot.service
sudo systemctl daemon-reload
sudo systemctl enable tradingbot.service
sudo systemctl restart tradingbot.service

echo "=============================================================================="
echo "✅ DESPLIEGUE EN VM DE GOOGLE CLOUD COMPLETADO EXITOSAMENTE"
echo "Para verificar el estado del bot en vivo ejecuta: sudo systemctl status tradingbot"
echo "Para ver los logs en tiempo real ejecuta: tail -f /opt/trading-bot-pro/trading_bot.log"
echo "=============================================================================="

