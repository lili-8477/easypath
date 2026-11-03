FROM python:3.11-slim
WORKDIR /srv/easypath
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY ann_common.py ann_db.py ./
COPY app ./app
# data is mounted, never baked into the image
ENV EASYPATH_RESULTS=/data/results PORT=8765
EXPOSE 8765
WORKDIR /srv/easypath/app
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers"]
