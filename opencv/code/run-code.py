from trainer import Trainer
import numpy as np

if __name__ == "__main__":
    TRAINER = Trainer()
    # TRAINER.resize('/input', '/output')
    # TRAINER.train('/output')
    TRAINER.load_model('/output/model/model_bowling.tflearn')
    predicted = TRAINER.predict_image('/input/01010/bowling02101.jpg')
                
    if np.argmax(predicted[0]) == 0:
        print ("It's a full rack")
        print (predicted[0])
    else:
        print ("It's not a full rack")
        print (predicted[0])