FROM continuumio/anaconda
EXPOSE 5000

RUN conda install opencv
RUN pip install imutils tensorflow tflearn sklearn

ADD ./input /input
ADD ./code /code
WORKDIR code

CMD ["python", "run-code.py"]