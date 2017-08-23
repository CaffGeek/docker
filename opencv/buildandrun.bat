docker build . -t caffgeek/opencv
docker run --rm -it -v /c/code/docker/opencv/output:/output --name opencv caffgeek/opencv 