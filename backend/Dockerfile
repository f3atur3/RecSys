FROM python:3.10

RUN mkdir /application

WORKDIR /application

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

ENTRYPOINT [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80" ]