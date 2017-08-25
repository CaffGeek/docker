FROM continuumio/anaconda
EXPOSE 5000

RUN conda install opencv
RUN pip install imutils tensorflow tflearn sklearn

ADD ./code /code
ADD ./training /training
ADD ./model /model
ADD ./test /test

WORKDIR code

CMD ["python", "run-code.py"]