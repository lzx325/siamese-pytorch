

import os
import sys
import time
import pickle
from collections import deque,OrderedDict
from os.path import join,basename,dirname,splitext
from pprint import pprint
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import yaml

import torch
from torch import tensor
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms

import tensorboardX

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from mydataset import OmniglotTrain, OmniglotTest, OmniglotStaticDataset, ICDARDataset
import model
import config
import contrastive

def save_model(net,ckpt_path):
    if isinstance(net,nn.DataParallel):
        torch.save(net.module.state_dict(),ckpt_path)
    else:
        torch.save(net.state_dict(),ckpt_path)

def load_model(net,ckpt_path):
    assert not isinstance(net,nn.DataParallel)
    state_dict=torch.load(ckpt_path)
    new_state_dict=OrderedDict()
    for key,value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key.replace("module.","")]=value
        else:
            new_state_dict[key]=value
    net.load_state_dict(new_state_dict)

def to_device(net,try_multiple_gpus=True,device_ids=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=net.to(device)
    if try_multiple_gpus and torch.cuda.device_count()>1:
        if device_ids is None:
            device_ids=list(range(torch.cuda.device_count()))
        net=nn.DataParallel(net,device_ids=device_ids)
    return net

class Trainer(object):
    def __init__(
        self,
        net,
        args
    ):
        self.net=net
        if args.loss_fn=="BCEWithLogits":
            self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
            self.model_return_vectors=False

        elif args.loss_fn=="Contrastive":
            self.loss_fn = contrastive.ContrastiveLoss()
            self.model_return_vectors=True

        # Declare Optimizer
        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        self.args=args
        self.ckpt_dir=join(args.train_dir,"ckpt")
        os.makedirs(self.ckpt_dir,exist_ok=True)
        self.tensorboard_dir=join(args.train_dir,"tb")
        self.writer=tensorboardX.SummaryWriter(self.tensorboard_dir)

    def train(self,train_dataset,eval_dataset):
        args=self.args
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        device=next(self.net.parameters()).device
        self.net.train()
        print(f"start evaluating epoch 0:")
        print("train set:")
        scores=self.evaluate(train_dataset,max_eval_steps=args.max_eval_steps,shuffle=True)
        auc=scores["auc"]
        loss=scores["loss"]
        print(f"auc: {auc}, loss: {loss}")
        self.writer.add_scalar("eval-train/auc",auc, 0)
        self.writer.add_scalar("eval-train/loss",loss, 0)

        print("eval set:")
        scores=self.evaluate(eval_dataset)
        auc=scores["auc"]
        loss=scores["loss"]
        print(f"auc: {auc}, loss: {loss}")
        self.writer.add_scalar("eval-test/auc",auc, 0)
        self.writer.add_scalar("eval-test/loss",loss, 0)
        for epoch in range(1, args.epochs+1):
            print(f"start training epoch {epoch}")
            for i, data in enumerate(train_dataloader):
                img0, img1, label = data
                img0, img1, label = img0.to(device), img1.to(device), label.to(device)
                self.optimizer.zero_grad()
                if args.loss_fn=="BCEWithLogits":
                    pred = self.net(img0, img1,self.model_return_vectors)
                    loss = self.loss_fn(pred, label)
                elif args.loss_fn=="Contrastive":
                    out1, out2=self.net(img0,img1,self.model_return_vectors)
                    loss=self.loss_fn(out1,out2,label)
                
                print(f"step {i+1}/{len(train_dataloader)}, loss {loss.item()}")
                self.writer.add_scalar("train/loss",loss,(epoch-1)*len(train_dataloader)+i, summary_description=f"{len(train_dataloader)} steps per epoch")
                loss.backward()
                self.optimizer.step()

            print(f"start evaluating epoch {epoch}:")
            print("train set:")
            scores=self.evaluate(train_dataset,max_eval_steps=args.max_eval_steps,shuffle=True)
            auc=scores["auc"]
            loss=scores["loss"]
            print(f"auc: {auc}, loss: {loss}")
            self.writer.add_scalar("eval-train/auc",auc, epoch)
            self.writer.add_scalar("eval-train/loss",loss, epoch)

            print("eval set:")
            scores=self.evaluate(eval_dataset)
            pprint({k:scores[k] for k in ["auc","loss"]})

            auc=scores["auc"]
            loss=scores["loss"]
            print(f"auc: {auc}, loss: {loss}")
            self.writer.add_scalar("eval-test/auc",auc, epoch)
            self.writer.add_scalar("eval-test/loss",loss, epoch)

            print("saving ckpt")
            save_model(self.net,join(self.ckpt_dir,f"epoch-{epoch:03d}.pt"))

    def evaluate(self,dataset,return_pred=False,max_eval_steps=None,shuffle=False):
        args=self.args
        eval_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers)
        device=next(self.net.parameters()).device
        loss=0.0
        pred_list=list()
        labels_list=list()
        self.net.eval()
        with torch.no_grad():
            for i,(img1,img2,label) in enumerate(eval_dataloader):
                if i%10==0:
                    print(f"step {i}/{len(eval_dataloader)}")
                img1,img2,label=img1.to(device),img2.to(device),label.to(device)
                
                if args.loss_fn=="BCEWithLogits":
                    pred=self.net(img1,img2,self.model_return_vectors)
                    loss_current = self.loss_fn(pred, label)
                elif args.loss_fn=="Contrastive":
                    out1, out2=self.net(img1,img2,self.model_return_vectors)
                    pred = F.pairwise_distance(out1, out2)
                    loss_current=self.loss_fn(out1,out2,label)
                labels_list.append(label.cpu().numpy())
                pred_list.append(pred.cpu().numpy())
                if type(max_eval_steps)==int:
                    loss+=loss_current.item()*len(label)/min(len(dataset),max_eval_steps*args.batch_size)
                    if i>max_eval_steps:
                        break
                else:
                    loss+=loss_current.item()*len(label)/len(dataset)


        preds=np.concatenate(pred_list)
        labels=np.concatenate(labels_list)
        auc=roc_auc_score(labels,preds)

        if return_pred:
            return {
                'auc':auc,
                'loss':loss,
                'preds':preds,
                'labels':labels,
            }
        else:
            return {
                'auc':auc,
                'loss':loss,
            }
    def __del__(self):
        self.writer.close()
        print("trainer finalized")


if __name__ == '__main__':


    SEED=0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED+1)
    
    args=config.parse_args()
    if args.mode=="train_original":
        data_transforms = transforms.Compose([
            transforms.RandomAffine(15),
            transforms.ToTensor()
        ])
        trainSet = OmniglotTrain(args.train_data_dir, transform=data_transforms)
        testSet = OmniglotTest(args.eval_data_dir, transform=transforms.ToTensor(), times = args.times, way = args.way)

        trainLoader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        testLoader = DataLoader(testSet, batch_size=args.way, shuffle=False, num_workers=args.workers)

        net = model.Siamese()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net=net.to(device)

        trainer=Trainer(net,args)

        # trainer.train()


        train_loss = []
        loss_val = 0
        time_start = time.time()
        queue = deque(maxlen=20)

        for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
            if batch_id > args.max_iter:
                break
            # img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
            # optimizer.zero_grad()
            # output = net.forward(img1, img2)
            # loss = loss_fn(output, label)
            # loss_val += loss.item()
            # loss.backward()
            # optimizer.step()
            # if batch_id % Flags.show_every == 0 :
            #     print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s'%(batch_id, loss_val/Flags.show_every, time.time() - time_start))
            #     loss_val = 0
            #     time_start = time.time()
            # if batch_id % Flags.save_every == 0:
            #     torch.save(net.state_dict(), Flags.model_path + '/model-inter-' + str(batch_id+1) + ".pt")
            if batch_id % args.test_every == 0:
                right, error = 0, 0
                for _, (test1, test2) in enumerate(testLoader, 1):
                    test1, test2 = test1.cuda(), test2.cuda()
                    test1, test2 = Variable(test1), Variable(test2)
                    output = net.forward(test1, test2).data.cpu().numpy()
                    pred = np.argmax(output)
                    if pred == 0:
                        right += 1
                    else: error += 1
                print('*'*70)
                print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
                print('*'*70)
                queue.append(right*1.0/(right+error))
            train_loss.append(loss_val)
        #  learning_rate = learning_rate * 0.95

        # with open('train_loss', 'wb') as f:
        #     pickle.dump(train_loss, f)

        acc = 0.0
        for d in queue:
            acc += d
        print("#"*70)
        print("final accuracy: ", acc/20)
    elif args.mode=="train":
        args.train_dir=join(args.train_dir,args.exp_code,dt.now().strftime('%Y-%m-%d-%H:%M:%S'))
        config_dir=join(args.train_dir,"config")
        os.makedirs(config_dir,exist_ok=True)
        with open(join(config_dir,"config.yaml"),'w') as f:
            yaml.dump(args.__dict__,f,sort_keys=False)
        if args.dataset=="Omniglot":
            data_transforms = transforms.Compose([
                transforms.RandomAffine(15),
                transforms.ToTensor()
            ])
            trainSet = OmniglotStaticDataset(
                args.train_summary_csv_fp,
                args.train_pairs_csv_fp,
                
                transform=data_transforms
            )
            testSet = OmniglotStaticDataset(
                args.eval_summary_csv_fp,
                args.eval_pairs_csv_fp,
                transform=data_transforms
            )
        elif args.dataset=="ICDAR":
            trainSet = ICDARDataset(
                args.training_data_csv,
                args.training_data_dir,
                transform=transforms.Compose(
                    [transforms.Resize((105, 105)), transforms.ToTensor()]
                ),
            )

            testSet = ICDARDataset(
                args.testing_data_csv,
                args.testing_data_dir,
                transform=transforms.Compose(
                    [transforms.Resize((105, 105)), transforms.ToTensor()]
                ),
            )
        modelclass=getattr(model,args.model_class)
        net = modelclass()
        net = to_device(net)
        trainer=Trainer(net,args)
        trainer.train(trainSet,testSet)

    elif args.mode=="eval":
        if args.dataset=="Omniglot":
            data_transforms = transforms.Compose([
                transforms.RandomAffine(15),
                transforms.ToTensor()
            ])
            trainSet = OmniglotStaticDataset(
                args.train_summary_csv_fp,
                args.train_pairs_csv_fp,
                
                transform=data_transforms
            )
            testSet = OmniglotStaticDataset(
                args.eval_summary_csv_fp,
                args.eval_pairs_csv_fp,
                transform=data_transforms
            )
        elif args.dataset=="ICDAR":
            trainSet = ICDARDataset(
                args.training_data_csv,
                args.training_data_dir,
                transform=transforms.Compose(
                    [transforms.Resize((105, 105)), transforms.ToTensor()]
                ),
            )

            testSet = ICDARDataset(
                args.testing_data_csv,
                args.testing_data_dir,
                transform=transforms.Compose(
                    [transforms.Resize((105, 105)), transforms.ToTensor()]
                ),
            )
        modelclass=getattr(model,args.model_class)
        net = modelclass()

        for fn in args.eval_ckpt_fn:
            ckpt_fp=join(args.eval_ckpt_dir,fn)
            load_model(net,ckpt_fp)
            net = to_device(net,try_multiple_gpus=args.multi_gpu)
            trainer=Trainer(net,args)
            scores=trainer.evaluate(testSet,return_pred=True)
            print("auc",scores["auc"])
            print("loss",scores["loss"])
    elif args.mode=="debug":
        modelclass=getattr(model,args.model_class)
        net = modelclass()
        trainer=Trainer(net,args)

        