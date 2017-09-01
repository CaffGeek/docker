#!/bin/sh
echo 'Testing files...'

python -m label_image --graph=tf_files/retrained_graph.pb  --labels=tf_files/retrained_labels.txt --image=/images/bowling/11111.1.jpg

echo 'Done'