#!/usr/bin/env bash
# Salir si hay error
set -o errexit

# Instalar requerimientos
pip install -r requirements.txt

# Recolectar estáticos
python manage.py collectstatic --no-input

# Migraciones (opcional si usas mongoengine/djongo puro a veces no se necesita, pero mejor dejarlo)
python manage.py migrate