from trainer import Trainer
import numpy as np

from colorama import init
from colorama import Fore, Back, Style
init(autoreset=True)

class App(object):
    def __init__(self):
        self.trainer = Trainer()
        self.trainer.resize('/input', '/output')
        self.trainer.train('/output')
        #self.trainer.load_model('/model/model_bowling.tflearn')

    def testImage(self, image, expectedLabel):
        predicted = self.trainer.predict_image(image)
        actualLabel = np.argmax(predicted[0])
        if actualLabel == expectedLabel:
            print ("{}PASS! Expected {} predicted {} in {}".format(Fore.GREEN, expectedLabel, actualLabel, image))
        else:
            print ("{}FAIL! Expected {} predicted {} in {}".format(Back.RED, expectedLabel, actualLabel, image))

app = App()
print(app.trainer.tf_image_labels)
app.testImage('/input/01010/bowling02101.jpg', 2)
app.testImage('/input/11100/bowling00701.jpg', 6)
app.testImage('/input/00000/bowling01051.jpg', 0)
app.testImage('/input/11111/bowling00101.jpg', 7)
app.testImage('/misc/dog.jpg', 8)
app.testImage('/misc/blue.png', 8)
app.testImage('/misc/black.png', 8)