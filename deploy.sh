#!/bin/bash
# Actualiza el dashboard estatico (web/) con la data mas reciente de la base
# de equipos, y sube todo a GitHub -- Netlify y Streamlit Cloud redeploy
# automaticamente al detectar el push (una vez conectados al repo).
#
# Uso:
#   ./deploy.sh "mensaje del commit"
#   ./deploy.sh                          # usa un mensaje por defecto con la fecha

set -e
cd "$(dirname "$0")"

MSG="${1:-Actualizacion $(date +%Y-%m-%d\ %H:%M)}"

echo "1/3  Regenerando JSON del dashboard estatico (export_web_data.py)..."
.venv/bin/python export_web_data.py

echo
echo "2/3  Preparando commit..."
git add -A
if git diff --cached --quiet; then
  echo "     Sin cambios que subir."
  exit 0
fi
git commit -m "$MSG"

echo
echo "3/3  Subiendo a GitHub (dispara redeploy de Netlify + Streamlit Cloud)..."
git push

echo
echo "Listo. Netlify y Streamlit Cloud deberian empezar a redesplegar en unos segundos."
