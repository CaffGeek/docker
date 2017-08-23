FROM continuumio/anaconda
EXPOSE 5000

RUN conda install opencv
RUN pip install imutils tensorflow tflearn sklearn

ADD ./code /code
ADD ./input /input
ADD ./model /model
ADD ./misc /misc

WORKDIR code

CMD ["python", "run-code.py"]