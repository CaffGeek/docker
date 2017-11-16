import label_image

from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Bowling classifier app v6!"

@app.route("/test")
def test():
    labels = label_image.load_labels('/tf/tf_files/retrained_labels.txt')
    label_image.load_graph('/tf/tf_files/retrained_graph.pb')
    image_data = label_image.load_image('/images/bowling/11111.1.jpg')
    return label_image.run_graph(image_data, labels, 'DecodeJpeg/contents:0', 'final_result:0', 5)

if __name__ == "__main__":
    app.run(host='0.0.0.0')