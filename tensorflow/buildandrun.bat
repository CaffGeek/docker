cls
docker build . -t caffgeek/tensorflow --squash
docker run --rm -it -p 5000:5000 --name tensorflow caffgeek/tensorflow