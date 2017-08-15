import numpy as np
import os
import cv2

def Gen_data(max_length):
    dirrr = "/home/edward/Workspace/Hoopcam/Code/Beta Testing/Data"
    train_examples = os.listdir(dirrr)

    train_input = []
    train_output = []
    test_input = []
    test_output = []
    for example in train_examples:

        labels = {'JS': 0, 'LU': 1, 'NN': 2}
        y = [0,0,0]
        y[labels[example[-2:]]] = 1
        train_output.append(y)

        example_path = os.path.join(dirrr,example)
        images = os.listdir(example_path)

        x = []
        for count,image in enumerate(images):
            img = os.path.join(example_path,image)
            img = cv2.imread(img)
            img = img.ravel()
            x.append(img)

        max_length = max_length    
        no_of_pads = max_length - count -1
        for i in range(no_of_pads):
            x.append(np.zeros(img.shape))
        
        x = np.array(x)
        train_input.append(x)

    train_input = np.array(train_input)
    train_output = np.array(train_output)

    test_input = train_input
    test_output = train_output
    return (train_input,train_output,test_input,test_output)


if __name__ == "__main__":
    train_input,train_output,test_input,test_output = Gen_data()
    print(train_input.shape)
    for i in train_input:
        print(i.shape)