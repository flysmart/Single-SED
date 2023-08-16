import torch
import torchvision
import torch.nn as nn
import numpy as np
import json
import utils
import validate
import argparse
import models.densenet
import models.resnet
import models.inception
import models.bilstm
import models.cnn
import models.crnn
import models.crnn1
import models.transformer
from models.conformer import model
import models.septr
import dataloaders.datasetaug
import dataloaders.datasetnormal

from tqdm import tqdm
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)


def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(data_loader)) as t:
        for batch_idx, data in enumerate(data_loader):
            inputs = data[0].to(device)
            target = data[1].squeeze(1).to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    return loss_avg()

def train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, writer, params, split, scheduler=None):
    best_acc = 0.0
    train_losses = []
    valid_accuracy = []
    valid_f1 = []
    for epoch in range(params.epochs):
        avg_loss = train(model, device, train_loader, optimizer, loss_fn)

        acc, f1, cm = validate.evaluate(model, device, val_loader)
        print("Epoch {}/{} Loss:{} Valid Acc:{} Valid F1:{}".format(epoch+1, params.epochs, avg_loss, acc, f1))

        train_losses.append(avg_loss)
        valid_accuracy.append(acc)
        valid_f1.append(f1)
        if (epoch+1) % 1 == 0:
            np.save("train_losses_{}.npy".format(split), train_losses)
            np.save("valid_accuracy_{}.npy".format(split), valid_accuracy)
            np.save("valid_f1_{}.npy".format(split), valid_f1)
            cm.to_csv('confusion_matrix_{}.csv'.format(split), index=False)

        is_best = (acc > best_acc)
        if is_best:
            best_acc = acc
        if scheduler:
            scheduler.step()

        #utils.save_checkpoint({"epoch": epoch + 1,
        #                       "model": model.state_dict(),
        #                       "optimizer": optimizer.state_dict()}, is_best, split, "{}".format(params.checkpoint_dir))
        writer.add_scalar("data{}/trainingLoss{}".format(params.dataset_name, split), avg_loss, epoch)
        writer.add_scalar("data{}/valLoss{}".format(params.dataset_name, split), acc, epoch)
    writer.close()


if __name__ == "__main__":
    args = parser.parse_args()
    params = utils.Params(args.config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(1, params.num_folds+1):
        if params.dataaug:
            train_loader = dataloaders.datasetaug.fetch_dataloader( "{}training128mel{}.pkl".format(params.data_dir, i), params.dataset_name, params.batch_size, params.num_workers, 'train')
            val_loader = dataloaders.datasetaug.fetch_dataloader("{}validation128mel{}.pkl".format(params.data_dir, i), params.dataset_name, params.batch_size, params.num_workers, 'validation')
        else:
            train_loader = dataloaders.datasetnormal.fetch_dataloader( "{}training128mel{}.pkl".format(params.data_dir, i), params.dataset_name, params.batch_size, params.num_workers)
            val_loader = dataloaders.datasetnormal.fetch_dataloader("{}validation128mel{}.pkl".format(params.data_dir, i), params.dataset_name, params.batch_size, params.num_workers)

        writer = SummaryWriter(comment=params.dataset_name)
        if params.model=="densenet":
            model = models.densenet.DenseNet(params.dataset_name, params.pretrained).to(device)
        elif params.model=="resnet":
            model = models.resnet.ResNet(params.dataset_name, params.pretrained).to(device)
        elif params.model=="inception":
            model = models.inception.Inception(params.dataset_name, params.pretrained).to(device)
        elif params.model=="bilstm":
            model = models.bilstm.BiLSTM(params.input_size, params.hidden_size,params.num_layer,params.num_classes).to(device)
        elif params.model=="crnn":
            model = models.crnn.CRNN(params.hidden_size,params.num_layer,params.num_classes).to(device)
        elif params.model=="transformer":
            model = models.transformer.AudioTransformer(params.input_size, params.hidden_size,params.num_layer,params.num_classes).to(device)
        elif params.model=="conformer":
            model = models.conformer.model.AudioConformer().to(device)
        elif params.model=="septr":
            model = models.septr.SeparableTr().to(device)
        elif params.model=="crnn1":
            model = models.crnn1.CRNN(params.num_chanel, params.hidden_size,params.num_layer,params.num_classes).to(device)
        elif params.model == "cnn":
            model = models.cnn.CNN(params.num_chanel,  params.num_classes).to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

        if params.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
        else:
            scheduler = None

        train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, writer, params, i, scheduler)


