rem docker run -it --publish 6006:6006 --volume /c/code/docker/tensorflow/tf_files:/tf_files --workdir /tf_files tensorflow/tensorflow:1.1.0 bash

docker build . -t caffgeek/tensorflow
docker run --rm -it -v /c/code/docker/tensorflow/tf_files:/tf_files --name tensorflow caffgeek/tensorflow