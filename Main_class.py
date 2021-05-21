import torch

import Data_Setup
from pathlib import Path
import Train
import Test
import matplotlib.pyplot as plt

def main():
    print("Press 1: Male Dataset  Press 2: Female Dataset Press 3: Mixed Dataset")
    inp = input()
    if inp == "1":
        dataset_path = Path('Face Mask Dataset Male/Train')
        pickle_path = Data_Setup.load_data(dataset_path)
    elif inp == "2":
        dataset_path = Path('Face Mask Dataset Female/Train')
        pickle_path = Data_Setup.load_data(dataset_path)
    elif inp == "3":
        dataset_path = Path('Face Mask Dataset Mixed/Train')
        pickle_path = Data_Setup.load_data(dataset_path)
    # pickle_path = Path('Face Mask Dataset/Train/dataset.pickle')

    # dataset_path = Path('Face Mask Dataset Male/Train')
    # #dataset_path = Path('Face Mask Dataset Female/Train')
    # #dataset_path = Path('Face Mask Dataset Mixed/Train')
    pickle_path = Data_Setup.load_data(dataset_path)

    #pickle_path = Path('Face Mask Dataset/Train/dataset.pickle')

    #train_data, test_data = Data_Setup.prepare_data(pickle_path)
    train_data, test_data = Data_Setup.prepare_data_kfold(pickle_path)
    #train_data_size = len(train_data)
    print("Data Preprocessing done..")
    model = Train.FaceMaskDetectorCNN()
    i=0
    for td in train_data:
        i+=1
        Train.train_model(td[0],model,len(td[0]),i)
    print('Training has finished')
    print("Accuracy for each K-Fold:")
    for acc in Train.k_fold_accuracy:
        print(acc)
    print("Training accuracy {:.2f}%".format(sum(Train.aggregate)/10))
    model.eval()


    while True:
        print("Press 1:test set  2: single image 9: exit")
        i = input()
        if i == "1":

            test_data_size = len(test_data[0])
            Test.test_model(test_data[0], test_data_size,model )

            #Test.test_model(test_data, test_data_size, model)
            #plt.plot(Train.ratio)
            #plt.xlabel("epochs")
            #plt.ylabel("loss")
            plt.title('Confusion Matrix')
            plt.show()
            continue
        elif i == "2":
            print("Provide image path")
            path = input()
            test_image = Data_Setup.load_prepare_test_image(path)
            Test.test_model(test_image,1,model)
        else:
            break

if __name__ == "__main__":
    main()
