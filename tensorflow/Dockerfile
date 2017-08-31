FROM tensorflow/tensorflow:1.1.0
EXPOSE 6006

ADD ./tf_files /tf_files
ADD ./code /code

WORKDIR /code

#RUN python retrain.py --bottleneck_dir=bottlenecks --how_many_training_steps=500 --model_dir=models/ --summaries_dir=training_summaries/bowling --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir=images

CMD ["bash", "test-files.sh"]