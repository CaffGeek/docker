FROM continuumio/anaconda
EXPOSE 5000

RUN conda install opencv
RUN pip install imutils

ADD ./code /code
WORKDIR code

CMD ["python", "run-code.py"]