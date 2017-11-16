import label_image

from flask import Flask, request
app = Flask(__name__)

@app.route("/")
def uploadForm():
    return '''
        <form action="" method="post" enctype="multipart/form-data">
            <input type="file" name="imagefile" id="imagefile"/>
            <input type="submit" />
        </form>'''

@app.route("/", methods=['POST'])
def upload():
    labels = label_image.load_labels('/tf/tf_files/retrained_labels.txt')
    label_image.load_graph('/tf/tf_files/retrained_graph.pb')

    print "request.files['imagefile']:", request.files['imagefile']
    image_data = request.files['imagefile'].stream.read()
    
    results = label_image.run_graph(image_data, labels, 'DecodeJpeg/contents:0', 'final_result:0', 5)
    return '<pre>' + results + '</pre>'

@app.route("/version")
def version():
    return "Bowling Classifier app v6!"

if __name__ == "__main__":
    app.run(host='0.0.0.0')