import torch
import torchvision
import os
import cv2 as cv
import re
from af2 import STPN

def test_model(model, inp):
    labels = {0: 'Archery', 1: 'Kayaking'}
    model.eval()    # Set model to evaluate mode

    f1t, f2t, f3t, rgb = [], [], [], [1, 1]
    f1t.extend(([], []))
    f2t.extend(([], []))
    f3t.extend(([], []))

    with torch.no_grad():
        for frame in sorted(os.listdir(inp)):
            img = cv.imread(os.path.join(inp, frame))
            tensor = torch.from_numpy(img).cuda()
            tensor = tensor.float()
            num = int(re.search(r'\d+', frame).group())
            cat = frame.split(str(num))[0]
            if cat == "flow":
                if num <= 10:
                    f1t[0].append(tensor)
                elif num <= 20:
                    f2t[0].append(tensor)
                elif num <= 30:
                    f3t[0].append(tensor)
            elif cat == "flip":
                if num <= 10:
                    f1t[1].append(tensor)
                elif num <= 20:
                    f2t[1].append(tensor)
                elif num <= 30:
                    f3t[1].append(tensor)
            else:
                if num == 1:
                    rgb[0] = tensor
                else:
                    rgb[1] = tensor

        flow1 = torch.stack([torch.stack(f1t[0]).cuda(), torch.stack(f1t[1]).cuda()]).cuda()
        flow1 = flow1.permute(0, 1, 4, 3, 2)
        flow2 = torch.stack([torch.stack(f2t[0]).cuda(), torch.stack(f2t[1]).cuda()]).cuda()
        flow2 = flow2.permute(0, 1, 4, 3, 2)
        flow3 = torch.stack([torch.stack(f3t[0]).cuda(), torch.stack(f3t[1]).cuda()]).cuda()
        flow3 = flow3.permute(0, 1, 4, 3, 2)
        rgb = torch.stack(rgb).cuda()
        rgb = rgb.permute(0, 3, 2, 1)
        
        outputs = model(rgb[0].unsqueeze_(0), flow1[0], flow2[0], flow3[0])
        outputs.unsqueeze_(0)
        _, preds = torch.max(outputs, 1, keepdim = True)
        print(labels[preds.item()])

    return labels[preds.item()]

if __name__ == '__main__':
    # Statistics: Accuracy calculations
    corrects = 0
    total = 0

    # Path containing trained model
    PATH = '/notebooks/storage/model2_1234.pt'
    # Load the model
    model = torch.load(PATH)

    # Path of folder being tested
    root = '/notebooks/storage/dataset/ucf3_post_split/test/Archery'
#     root = '/notebooks/storage/dataset/ucf3_post_split/test/Kayaking'

    doc = open('/notebooks/storage/results2_1234.txt', 'a')     # Record the results in a text document
    for vids in sorted(os.listdir(root)):
        total += 1
        inp = os.path.join(root, vids)
        print("Input: {}".format(inp))
        result = test_model(model, inp)

        if result == root.split('/')[-1]:
            corrects += 1
        doc.write("Input: {}\nResult: {}\n\n".format(inp, result))
    print("Results: {}/{} = {:4f}".format(corrects, total, corrects/total))
    doc.write("Results: {}/{} = {:4f}".format(corrects, total, corrects/total))
    doc.close()
