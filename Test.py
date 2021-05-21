import torch
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


def val_dataloader(validate_df,size) -> DataLoader:
    return DataLoader(validate_df, batch_size=size)


def test_model(test_data,size,model):
    print("Testing ",size, " images:")
    with torch.no_grad():
        test_correct = 0
        for i, data in enumerate(val_dataloader(test_data,size), 0):
            inputs, labels = data['image'], data['class']
            labels = labels.flatten()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            #total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    #print("Actual class:", labels, "Predicted class :", predicted)
    labels_n=[]
    predicted_n=[]
    #labels_n = ["With Mask" for x in labels if x == torch.tensor(0)]
    for i,j in zip(labels,predicted):
        if i== torch.tensor(0):
            labels_n.append("With Mask")
        elif i==torch.tensor(1):
            labels_n.append("With out Mask")
        elif i==torch.tensor(2):
            labels_n.append("Not a person")
        if j==torch.tensor(0):
            predicted_n.append("With Mask")
        elif j==torch.tensor(1):
            predicted_n.append("With out Mask")
        elif j==torch.tensor(2):
            predicted_n.append("Not a person")
    titles = ['Actual class', 'Predicted class']
    data = [titles] + list(zip(labels_n, predicted_n))
    for i, d in enumerate(data):
        line = '|'.join(str(x).ljust(20) for x in d)
        print(line)
        if i == 0:
            print('-' * len(line))


    #cm_display.show()
    print('Test Accuracy: {} %'.format((test_correct / size) * 100))
    if size>1:
        target_names = ['With Mask', 'Without Mask', 'Not a Person']
        print("Classification Report for the test data")
        print(classification_report(labels, predicted, target_names=target_names))
        cm = confusion_matrix(labels_n, predicted_n)
        ConfusionMatrixDisplay(cm).plot()

