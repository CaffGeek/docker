docker build . -t caffgeek/opencv
docker run --rm -it -v /c/code/docker/opencv/output:/output -v /c/code/docker/opencv/model:/model --name opencv caffgeek/opencv 