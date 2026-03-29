FROM python:3.12-slim
WORKDIR /app
COPY conversational/app/ ./conversational/app/
EXPOSE 8080
CMD ["python3", "conversational/app/server.py"]
