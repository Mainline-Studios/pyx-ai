# gthread + threads: avoid one long stream blocking all traffic (default sync worker = single request at a time).
# Timeout 300s for long streamed /talk replies. $PORT is set by Cloud Run / buildpacks.
web: gunicorn app:app --bind 0.0.0.0:$PORT --worker-class gthread --workers 1 --threads 16 --timeout 300 --graceful-timeout 60 --keep-alive 75
