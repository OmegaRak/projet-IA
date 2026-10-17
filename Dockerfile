FROM python:3.12-slim

WORKDIR /srv/api

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY models /srv/models

ENV MODEL_DIR=/srv/models
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
