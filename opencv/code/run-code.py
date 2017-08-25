from trainer import Trainer
import numpy as np

from colorama import init
from colorama import Fore, Back, Style
init(autoreset=True)

class App(object):
    def __init__(self):
        self.trainer = Trainer()
        self.trainer.resize('/training', '/output')
        self.trainer.train('/output')
        #self.trainer.load_model('/model/model_bowling.tflearn')

    def testImage(self, image, expectedLabel):
        predicted = self.trainer.predict_image(image)
        actualLabel = self.trainer.labels[np.argmax(predicted[0])]        
        if actualLabel == expectedLabel:
            print ("{}PASS! Expected {} predicted {} in {}".format(Fore.GREEN, expectedLabel, actualLabel, image))
        else:
            print ("{}FAIL! Expected {} predicted {} in {}".format(Back.RED, expectedLabel, actualLabel, image))
            d = dict(zip(self.trainer.labels, np.round(predicted[0], 2)))
            for k in sorted(d):
                print k, d[k]

app = App()
app.testImage('/test/00001.PNG', '00001')
app.testImage('/test/11100.jpg', '11100')
app.testImage('/test/11111.jpg', '11111')
app.testImage('/test/dog.jpg', 'other')
app.testImage('/test/blue.png', 'other')
app.testImage('/test/black.png', 'other')