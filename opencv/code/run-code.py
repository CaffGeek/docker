from trainer import Trainer
import numpy as np

class App(object):
    def __init__(self):
        self.trainer = Trainer()
        self.trainer.resize('/input', '/output')
        self.trainer.train('/output')
        #self.trainer.load_model('/model/model_bowling.tflearn')

    def showResult(self, image):
        predicted = self.trainer.predict_image(image)
        if np.argmax(predicted[0]) == 0:
            print ("{} is a full rack".format(image))
        else:
            print ("{} is not a full rack".format(image))
        print (predicted[0])

app = App()
app.showResult('/input/01010/bowling02101.jpg')
app.showResult('/input/11100/bowling00701.jpg')
app.showResult('/input/00000/bowling01051.jpg')
app.showResult('/input/11111/bowling00101.jpg')
app.showResult('/misc/dog.jpg')
app.showResult('/misc/blue.png')
app.showResult('/misc/black.png')