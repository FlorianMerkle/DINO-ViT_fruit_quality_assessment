import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
import foolbox as fb
from sklearn.neural_network import MLPClassifier
from dino_experiments.util import get_embeddings, get_seeded_data_loader
from skorch import NeuralNetClassifier
from time import time


GLOBAL_SEED = 1
RESIZE = True
BATCH_SIZE = 8
TEST_SAMPLE_SIZE = VAL_SAMPLE_SIZE = TRAIN_SAMPLE_SIZE = -1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = 'fayoum'

def evaluate_advs(transformer,advs, clf):
    normalize_values = ((0.6237, 0.6076, 0.5367), (0.1759, 0.1925, 0.3011))
    normalized_advs = TF.normalize(advs, mean =normalize_values[0], std=normalize_values[1])
    test_loader = get_seeded_data_loader(dataset, "test", GLOBAL_SEED, resize=RESIZE, batch_size=len(normalized_advs), normalize=False)
    benign_labels = next(iter(test_loader))[1]
    adv_test_dataset = torch.utils.data.TensorDataset(normalized_advs, benign_labels)
    adv_test_dataloader = torch.utils.data.DataLoader(adv_test_dataset, batch_size=10)
    adv_embeddings, ys = get_embeddings(transformer, adv_test_dataloader,TEST_SAMPLE_SIZE)
    acc = clf.score(adv_embeddings, ys)
    return acc

def get_torch_classifier():
    
    try:
        m = torch.load('/home/florian/dev/DINO-ViT_fruit_quality_assessment/adversarial_attacks/saved-models/torch-classifier.pkl')
    except:
        print('no model to load found')
        train_loader = get_seeded_data_loader(dataset, "train", GLOBAL_SEED, resize=RESIZE, batch_size=BATCH_SIZE)
        val_loader = get_seeded_data_loader(dataset, "val", GLOBAL_SEED, resize=RESIZE, batch_size=BATCH_SIZE)
        transformer = torch.hub.load('facebookresearch/dino:main', 'dino_vits8', verbose=False).to(device)
        m = DinoClassifier(transformer, 384, 4).cuda()
        for param in m.feature_extractor.parameters():
            param.requires_grad = False
        _fit(m, train_loader, val_loader, 25, device)
    test_loader = get_seeded_data_loader(dataset, "test", GLOBAL_SEED, resize=RESIZE, batch_size=BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    torch.save(m, '/home/florian/dev/DINO-ViT_fruit_quality_assessment/adversarial_attacks/saved-models/torch-classifier.pkl')

    accuracy, loss = _evaluate_model(m, test_loader, device, criterion)
    print('test acc:', accuracy,', test loss:' ,loss)
    return m,accuracy

def get_skorch_classifier_acc():
    test_loader = get_seeded_data_loader(dataset, "test", GLOBAL_SEED, resize=RESIZE, batch_size=60)
    train_loader = get_seeded_data_loader(dataset, "train", GLOBAL_SEED, resize=RESIZE, batch_size=176)
    val_loader = get_seeded_data_loader(dataset, "val", GLOBAL_SEED, resize=RESIZE, batch_size=60)
    X_train_only,y_train_only = next(iter(train_loader))
    X_val,y_val = next(iter(val_loader))
    X_train, y_train = np.concatenate((X_train_only, X_val)), np.concatenate((y_train_only, y_val))
    
    transformer = torch.hub.load('facebookresearch/dino:main', 'dino_vits8', verbose=False).to(device)
    m = DinoClassifier(transformer, 384, 4)
    for param in m.feature_extractor.parameters():
        param.requires_grad = False
    net = NeuralNetClassifier(
        m,
        max_epochs=30,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam,
        batch_size=8,
    )
    
    net.fit(X_train, y_train)
    test_loader = get_seeded_data_loader(dataset, "test", GLOBAL_SEED, resize=RESIZE, batch_size=7*8)
    X_test, y_test = next(iter(test_loader))
    acc = net.score(X_test, y_test)
    print(acc)
    return acc

def get_original_classifier():
    transformer = torch.hub.load('facebookresearch/dino:main', 'dino_vits8', verbose=False)
    transformer.to(device)
    
    test_loader = get_seeded_data_loader(dataset, "test", GLOBAL_SEED, resize=RESIZE, batch_size=BATCH_SIZE)
    X_test, y_test = get_embeddings(transformer, test_loader, TEST_SAMPLE_SIZE, batch_size=BATCH_SIZE)
    
    train_loader = get_seeded_data_loader(dataset, "train", GLOBAL_SEED, batch_size=BATCH_SIZE, resize=RESIZE)
    X_train, y_train = get_embeddings(transformer, train_loader, TRAIN_SAMPLE_SIZE, batch_size=BATCH_SIZE)
    
    clf = MLPClassifier(random_state=GLOBAL_SEED, max_iter=10000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = clf.score(X_test, y_test)
    
    return clf, acc


def get_adversarial_robustness(m, attack,eps, confidence=.02):
    test_loader = get_seeded_data_loader(dataset, "test", GLOBAL_SEED, resize=RESIZE, batch_size=4, normalize=False)
    
    m.eval()
    preprocessing = dict(mean=(0.6237, 0.6076, 0.5367), std=(0.1759, 0.1925, 0.3011), axis=-3)
    bounds = (0, 1)
    fmodel = fb.PyTorchModel(m, bounds=bounds,preprocessing=preprocessing)
    successful_advs = 0
    total_advs = 0
    advs = np.empty((0,3,224,224))
    if attack == 'PGD': att = fb.attacks.PGD()
    if attack == 'FGSM': att = fb.attacks.FGSM()
    if attack == 'CW': att = fb.attacks.L2CarliniWagnerAttack(confidence=confidence, binary_search_steps=5, steps=100, initial_const=1)
    if attack == 'DF': att = fb.attacks.L2DeepFoolAttack(overshoot=confidence)
    if attack == 'BB0': att = fb.attacks.L0BrendelBethgeAttack(init_attack=fb.attacks.LinearSearchBlendedUniformNoiseAttack(),overshoot=confidence, binary_search_steps=3, lr_num_decay=5, steps=50, lr=.01)
    total_time = 0
    for x_batch, y_batch in test_loader:
        t0=time()
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        raw, clipped, is_adv = att(fmodel, x_batch, y_batch, epsilons=eps)
        advs = np.concatenate((advs, clipped.cpu()))
        total_advs += len(is_adv)
        successful_advs += is_adv.sum()
        print('ASR:',(successful_advs/total_advs).item())
        print('time per batch:',time()-t0)
        total_time += time()-t0
    time_per_batch = total_time/len(test_loader)
    return 1-successful_advs/total_advs, advs,time_per_batch

def get_adversarial_robustness_l0(m, attack,lr, confidence=1.1):
    test_loader = get_seeded_data_loader(dataset, "test", GLOBAL_SEED, resize=RESIZE, batch_size=4, normalize=False)
    
    m.eval()
    preprocessing = dict(mean=(0.6237, 0.6076, 0.5367), std=(0.1759, 0.1925, 0.3011), axis=-3)
    bounds = (0, 1)
    fmodel = fb.PyTorchModel(m, bounds=bounds,preprocessing=preprocessing)
    successful_advs = 0
    total_advs = 0
    advs = np.empty((0,3,224,224))
    if attack == 'BB0': att = fb.attacks.L0BrendelBethgeAttack(init_attack=fb.attacks.LinearSearchBlendedUniformNoiseAttack(),overshoot=confidence, binary_search_steps=10, lr_num_decay=20, steps=1000, lr=lr)
    total_time = 0
    for x_batch, y_batch in test_loader:
        t0=time()
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        raw, clipped, is_adv = att(
            fmodel,
            x_batch,
            criterion=fb.criteria.Misclassification(y_batch),
            epsilons=None
        )
    
        advs = np.concatenate((advs, clipped.cpu()))
        total_advs += len(is_adv)
        successful_advs += is_adv.sum()
        print('ASR:',(successful_advs/total_advs).item())
        print('time per batch:',time()-t0)
        total_time += time()-t0
        torch.save(advs, f'./adversarial_attacks/adversarial-examples/l0_only{attack}-{lr}-{confidence}.pkl')
    time_per_batch = total_time/len(test_loader)
    return 1-successful_advs/total_advs, advs,time_per_batch
    


class DinoClassifier(nn.Module):
    def __init__(self, feature_extractor, num_features,num_classes):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(num_features, num_classes)
        self.train_stats=np.array([])
    def forward(self, inputs):
        x = F.relu(self.feature_extractor(inputs))
        return self.classifier(x)

def _fit(model, train_loader, val_loader, epochs, device, patience=None, evaluate_robustness=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    total_time = 0
    epochs_trained = 0
    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = [], [], [], []
    for epoch in range(epochs):  # loop over the dataset multiple times
        t0 = time()
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
        t1 = time()
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