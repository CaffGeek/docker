import label_image

labels = label_image.load_labels('/tf/tf_files/retrained_labels.txt')
label_image.load_graph('/tf/tf_files/retrained_graph.pb')

image_data = label_image.load_image('/images/bowling/11111.1.jpg')
label_image.run_graph(image_data, labels, 'DecodeJpeg/contents:0', 'final_result:0', 5)