from trainer import Trainer
import numpy as np

class App(object):
    def __init__(self):
        self.trainer = Trainer()
        self.trainer.resize('/input', '/output')
        self.trainer.train('/output')
        #self.trainer.load_model('/model/model_bowling.tflearn')

    def testImage(self, image, expectedLabel):
        predicted = self.trainer.predict_image(image)
        if np.argmax(predicted[0]) == expectedLabel:
            print ("PASS! {} {}".format(image, predicted[0]))
        else:
            print ("FAIL! {} {}".format(image, predicted[0]))

app = App()
print(app.trainer.tf_image_labels)
app.testImage('/input/01010/bowling02101.jpg', 1)
app.testImage('/input/11100/bowling00701.jpg', 1)
app.testImage('/input/00000/bowling01051.jpg', 1)
app.testImage('/input/11111/bowling00101.jpg', 0)
app.testImage('/misc/dog.jpg', 1)
app.testImage('/misc/blue.png', 1)
app.testImage('/misc/black.png', 1)