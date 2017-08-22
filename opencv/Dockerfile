FROM continuumio/anaconda
EXPOSE 5000

ADD ./code /code
WORKDIR code

RUN conda install opencv
RUN pip install imutils

CMD ["python", "run-code.py"]