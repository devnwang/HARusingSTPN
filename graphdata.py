import re
import numpy as np
from bokeh.plotting import *

output_file("graphs.html")

epoch = np.array([])
trainloss = np.array([])
valloss = np.array([])
trainacc = np.array([])
valacc = np.array([])
bestacc = np.array([])

lcnt = 0
acnt = 0

for line in open('log2.txt'):
#for line in open('testlog.txt'):
    # one epoch value per line
    epochnum = re.search('Epoch (\d+)', line)

    # one loss value per line
    # alternates train -> loss
    loss = re.search('Loss: (\d+\.\d+)', line)

    # two acc values per line
    # train, best -> val, best
    acc = re.findall('Acc: (\d+\.\d+)', line)


    if epochnum:
        #print('Epoch : ', epochnum.group(1))
        epoch = np.append(epoch, epochnum.group(1))
    if loss:
        if (lcnt == 0):
            trainloss = np.append(trainloss, loss.group(1))
            lcnt = lcnt + 1
        else:
            valloss = np.append(valloss, loss.group(1))
            lcnt = 0

        #print('Loss: ', loss.group(1))
    if acc:
        if (acnt == 0):
            trainacc = np.append(trainacc, acc[0])
            acnt = acnt + 1
        else:
            valacc = np.append(valacc, acc[0])
            bestacc = np.append(bestacc, acc[1])
            acnt = 0
        #print('Acc : ', acc[0])
        #print('Acc : ', acc[1])

epoch_max = np.max(epoch.astype(np.float))

loss_y_max = np.max(trainloss.astype(np.float)) if (np.max(trainloss.astype(np.float)) > np.max(valloss.astype(np.float))) else np.max(valloss.astype(np.float))
acc_y_max = np.max(trainacc.astype(np.float)) if (np.max(trainacc.astype(np.float)) > np.max(valacc.astype(np.float))) else np.max(valacc.astype(np.float))

loss_y_min = np.min(trainloss.astype(np.float)) if (np.min(trainloss.astype(np.float)) > np.min(valloss.astype(np.float))) else np.min(valloss.astype(np.float))
acc_y_min = np.min(trainacc.astype(np.float)) if (np.min(trainacc.astype(np.float)) > np.min(valacc.astype(np.float))) else np.min(valacc.astype(np.float))

train_acc_line = figure(title="Train/Val Accuracy/Epoch", x_axis_label='Epoch', y_axis_label='Accuracy', x_range = (0, epoch_max), y_range = (acc_y_min - 0.5, acc_y_max + 0.1))

train_acc_line.line(epoch, trainacc, legend="Train", line_width=1, color = 'navy', alpha = 0.5)
train_acc_line.line(epoch, valacc, legend="Validation", line_width=1, color = 'orange', alpha = 0.5)
train_acc_line.yaxis.major_label_orientation = "vertical"

train_loss_line = figure(title="Train/Val Loss/Epoch", x_axis_label='Epoch', y_axis_label='Loss', x_range = (0, epoch_max), y_range = (loss_y_min - 0.5, loss_y_max + 10))

train_loss_line.yaxis.major_label_orientation = "vertical"
train_loss_line.line(epoch, trainloss, legend="Train", line_width=1, color = 'navy', alpha = 0.5)
train_loss_line.line(epoch, valloss, legend="Validation", line_width=1, color = 'orange', alpha = 0.5)

p = gridplot([[train_acc_line, train_loss_line]], toolbar_location=None)

show(p)

print(np.max(trainloss.astype(np.float)))
print(np.max(valloss.astype(np.float)))

#print(trainloss)
#print(valloss)
#show(train_loss_line)
#print(epoch[40], ", " ,epoch[50], ", " ,epoch[60])
#print(trainacc[40], ", " ,trainacc[50], ", " ,trainacc[60])
#print(trainloss[40], ", " ,trainloss[50], ", " ,trainloss[60])
'''
print('Epoch = ',epoch)
print('Train Loss = ',trainloss)
print('Validation Loss = ',valloss)
print('Train Acc = ',trainacc)
print('Validation Acc = ',valacc)
print('Best Acc = ', bestacc)
'''