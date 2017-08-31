docker build . -t caffgeek/opencv
rem docker run --rm -it -v /c/code/docker/opencv/model:/model --name opencv caffgeek/opencv 
docker run --rm -it -v /c/code/docker/opencv/model:/model -v /c/code/docker/opencv/output:/output --name opencv caffgeek/opencv 