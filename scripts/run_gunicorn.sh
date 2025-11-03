source env/bin/activate

exec gunicorn api.api:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:5000 \
  --timeout 1500 \
  --log-level info