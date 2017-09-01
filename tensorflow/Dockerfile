FROM tensorflow/tensorflow:1.1.0
EXPOSE 5000

COPY ./images /images
COPY ./tf /tf

WORKDIR /tf
RUN python -m retrain --bottleneck_dir=tf_files/bottlenecks --model_dir=tf_files/models/ --summaries_dir=tf_files/training_summaries/bowling --output_graph=tf_files/retrained_graph.pb --output_labels=tf_files/retrained_labels.txt --image_dir=/images

RUN pip install Flask

#copy code and clean up windows line endings
COPY ./code /tf
RUN sed -i 's/\r//' /tf/*.*

#CMD ["python", "test-files.py"]
#CMD ["/bin/bash"]

ENTRYPOINT ["python"]
CMD ["app.py"]