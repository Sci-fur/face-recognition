#!/usr/bin/env bash
DIR="$(dirname "$(readlink -f "$0")")"
cd "$DIR" || exit 1
exec "$DIR/venv/bin/python" "$DIR/desktop_app.py"
