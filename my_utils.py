import torch
from torch import nn
import torch.nn.functional as F
import time
import torch.optim as optim


def _fit(model, train_loader, val_loader, epochs, device, patience=None, evaluate_robustness=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    total_time = 0
    epochs_trained = 0
    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = [], [], [], []
    for epoch in range(epochs):  # loop over the dataset multiple times
        t0 = time.time()
        acc_epoch_loss, avg_epoch_loss, epoch_accuracy, acc_epoch_accuracy = 0.0, 0.0, 0.0, 0.0
        
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            batchsize = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / batchsize
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

                # print statistics
            acc_epoch_loss += loss.item() 
            avg_epoch_loss = acc_epoch_loss / (i+1)
            acc_epoch_accuracy += accuracy
            avg_epoch_accuracy = acc_epoch_accuracy / (i+1)
            if i%10 == 0:
                print('[%d, %5d] loss: %.5f, train_accuracy: %.2f' %(epoch + 1, i + 1, loss.item(), accuracy))
        t1 = time.time()
        total_time += t1 - t0
        accuracy, loss = _evaluate_model(model, val_loader, device, criterion)
        #print('duration:', t1-t0,'- train loss: ',avg_epoch_loss,' - train accuracy: ',avg_epoch_accuracy,' - validation accuracy: ', accuracy,' - validation loss: ', loss)
        print('duration: %d s - train loss: %.5f - train accuracy: %.2f - validation loss: %.2f - validation accuracy: %.2f ' %(t1-t0, avg_epoch_loss, avg_epoch_accuracy, loss, accuracy))
        train_loss_hist.append(avg_epoch_loss)
        train_acc_hist.append(avg_epoch_accuracy)
        val_loss_hist.append(loss)
        val_acc_hist.append(accuracy)
        data = {
            'epoch': epoch+1,
            'train_loss':avg_epoch_loss, 
            'train_accuracy':avg_epoch_accuracy,
            'validation_loss':loss,
            'validation_accuracy':accuracy,
            'duration':total_time,
            'criterion':criterion,
            'optimizer':optimizer,
            'method': 'standard',
            'batchsize': len(next(iter(train_loader))[1])
        }
        
        
        #model.train_stats = model.train_stats.append(data, ignore_index=True)
        
        #if patience != None and patience < epoch and stop_early(val_loss_hist, patience) == True:
        #    epochs_trained = i + 1
        #    print('stopped early after', patience, 'epochs without decrease of validation loss')
        #    break
    print('Finished Training')
    
    return model.train_stats

def _evaluate_model(model, data_loader, device, criterion):
    correct = 0
    total = 0
    acc_loss = 0.0
    avg_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            #print(i)
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if criterion != None:
                loss = criterion(outputs, labels)
                acc_loss += loss.item() 
                avg_loss = acc_loss / (i+1)
            #print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    model.train()
    return accuracy, avg_loss