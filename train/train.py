import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import classification_report
from tabulate import tabulate
from tqdm import tqdm

from dataset import WakeWordDataset, collate_fn
from model import CNNModel

def save_checkpoint(checkpoint_path, model, optimizer, scheduler, model_params, notes=None):
    torch.save({
        "notes": notes,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }, checkpoint_path)

def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    acc = rounded_preds.eq(y.view_as(rounded_preds)).sum().item() / len(y)
    return acc

def train(train_loader, model, optimizer, loss_fn, device, epoch):
    print("\n starting train for epoch %s"%epoch)
    preds, losses, labels = [],[],[]

    for idx, (mfcc, label) in enumerate(train_loader):
        mfcc, label = mfcc.to('cpu'), label.to('cpu')
        optimizer.zero_grad()
        output = model(mfcc)
        
        loss = loss_fn(torch.flatten(output), torch.tensor(label, dtype = float))

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        pred = torch.sigmoid(output)
        preds += torch.flatten(torch.round(pred)).cpu()
        labels += torch.flatten(label).cpu()

        print("epoch: {}, Iter: {}/{}, loss: {}".format(epoch, idx, len(train_loader), loss.item()), end="\r")

    avg_train_loss = sum(losses)/len(losses)
    acc = binary_accuracy(torch.Tensor(preds), torch.Tensor(labels))
    print('avg train loss:', avg_train_loss, "avg train acc", acc)
    report = classification_report(torch.Tensor(labels).numpy(), torch.Tensor(preds).numpy())
    print(report)
    return acc, report

def valid(valid_loader, model, device, epoch):
    print("\n starting validation for epoch %s"%epoch)
    preds, losses, labels = [],[],[]

    for idx, (mfcc, label) in enumerate(valid_loader):
        with torch.no_grad():
            output = model(mfcc)
            loss = loss_fn(torch.Tensor(output), label)
            losses.append(loss.item())
            pred = torch.sigmoid(output)
            preds += torch.flatten(torch.round(pred)).cpu()
            labels += torch.flatten(label).cpu()

            print("epoch: {}, Iter: {}/{}, loss: {}".format(epoch, idx, len(valid_loader), loss.item()), end="\r")

        avg_train_loss = sum(losses)/len(losses)
        acc = binary_accuracy(torch.Tensor(preds), torch.Tensor(labels))
        print('avg val loss:', avg_train_loss, "avg val acc", acc)
        report = classification_report(torch.Tensor(labels).numpy(), torch.Tensor(preds).numpy())
        print(report)
        return acc, report
    
def main(args):
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = WakeWordDataset(json_data='json_training_dir/train_data.json')
    valid_data = WakeWordDataset(json_data='json_training_dir/test_data.json')

    train_data_loader = DataLoader(train_data,
                                   batch_size = 64,
                                   shuffle = True)
    valid_data_loader = DataLoader(valid_data,
                                   batch_size = 1,
                                   shuffle = True)
    
    model = CNNModel(num_classes=1)
    model = model.to(device)
    print("Loaded model", model)
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_train_acc, best_train_report = 0, None
    best_test_acc, best_test_report = 0, None
    best_epoch = 0

    for epoch in range(args.epochs):
        print("\nstarting training with learning rate", optimizer.param_groups[0]['lr'])
        train_acc, train_report = train(train_data_loader, model, optimizer, loss_fn, device, epoch)
        test_acc, test_report = valid(valid_data_loader, model, device, epoch)

        # record best train and test
        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        # saves checkpoint if metrics are better than last
        if args.save_checkpoint_path and test_acc >= best_test_acc:
            checkpoint_path = os.path.join(args.save_checkpoint_path, args.model_name + ".pt")
            print("found best checkpoint. saving model as", checkpoint_path)
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler,
                notes="train_acc: {}, test_acc: {}, epoch: {}".format(best_train_acc, best_test_acc, epoch),
            )
            best_train_report = train_report
            best_test_report = test_report
            best_epoch = epoch

        table = [["Train ACC", train_acc], ["Test ACC", test_acc],
                ["Best Train ACC", best_train_acc], ["Best Test ACC", best_test_acc],
                ["Best Epoch", best_epoch]]
        # print("\ntrain acc:", train_acc, "test acc:", test_acc, "\n",
        #     "best train acc", best_train_acc, "best test acc", best_test_acc)
        print(tabulate(table))

        scheduler.step(train_acc)

    print("Done Training...")
    print("Best Model Saved to", checkpoint_path)
    print("Best Epoch", best_epoch)
    print("\nTrain Report \n")
    print(best_train_report)
    print("\nTest Report\n")
    print(best_test_report)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_rate', type=int, default=8000, help='sample_rate for data')
    parser.add_argument('--epochs', type=int, default=100, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=32, help='size of batch')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='size of batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--model_name', type=str, default="cnn_wakeword", required=False, help='name of model to save')
    parser.add_argument('--save_checkpoint_path', type=str, default=None, help='Path to save the best checkpoint')
    parser.add_argument('--train_data_json', type=str, default=None, required=True, help='path to train data json file')
    parser.add_argument('--test_data_json', type=str, default=None, required=True, help='path to test data json file')
    parser.add_argument('--no_cuda', action='store_true', default=True, help='disables CUDA training')
    parser.add_argument('--num_workers', type=int, default=1, help='number of data loading workers')

    args = parser.parse_args()

    main(args)