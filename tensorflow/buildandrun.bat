cls
docker build . -t caffgeek/tensorflow
docker run --rm -it -p 5000:80 --name tensorflow caffgeek/tensorflow