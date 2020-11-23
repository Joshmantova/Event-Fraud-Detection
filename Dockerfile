FROM python:3.7

COPY ./requirements.txt /app/requirements.txt
COPY . /app

WORKDIR /app/flaskshell

RUN pip install -r ../requirements.txt

EXPOSE 8000

CMD python app.py