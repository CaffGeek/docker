import label_image

image_data = label_image.load_image('/images/bowling/11111.1.jpg')
labels = label_image.load_labels('tf_files/retrained_labels.txt')
label_image.load_graph('tf_files/retrained_graph.pb')

label_image.run_graph(image_data, labels, 'DecodeJpeg/contents:0', 'final_result:0', 5)