FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir poetry && poetry install --no-interaction --no-ansi
EXPOSE 8501
CMD ["poetry", "run", "streamlit", "run", "app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
