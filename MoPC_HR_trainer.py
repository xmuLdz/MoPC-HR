
# @Software: PyCharm
import copy

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import os

from torch.utils.data import Dataset,DataLoader,TensorDataset,ConcatDataset

from AIS100_1024 import AIS100_1024


from ais_resnet import resnet1d_ais
from ads_resnet import resnet1d_ads
import logging
class Trainer:
    def __init__(self, total_cls,init_num_classes,interval_nums):
        self.total_cls = total_cls

        self.batch_size = None

        self.init_num_classes = init_num_classes
        self.seen_cls=init_num_classes
        self.interval_nums = interval_nums
        self.dataset = AIS100_1024(total_cls,init_num_classes,interval_nums)
        # self.model = Resnet(32,init_num_classes,self.dataset.batch_num,total_cls).cuda()
        # self.model = resnet1d_ais().cuda()
        self.model = resnet1d_ads().cuda()

        model_params = list(self.model.parameters())
        self.num_layers = len(model_params)
        self.layer_lambdas = self.get_layer_lambdas(self.num_layers,1)

        self.batch_size = None

        self.prototype = None
        self.class_label = None

        self.init_accs = None
        self.previous_accs = []
        self.incremented_accs = []
        self.old_accs = []
        self.old_accs1=[]
        self.last_accs = []
        self.losses=[]
        self.initaccs=[]
        self.conf_matrix = torch.zeros(100, 100)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Solver total trainable parameters : ", total_params)
        print("---------------------------------------------")

    def get_layer_lambdas(self,num_layers,lambda_max):

        lambdas = []
        for l in range(1, num_layers + 1):
            lambda_l = lambda_max * (1 - (l - 1) / num_layers)
            lambdas.append(lambda_l)
        return lambdas

    def test(self, current_task):
        acces=[]
        for i in range(current_task+1):
            _, test_data = self.dataset.getNextClasses(i)
            test_loader = DataLoader(test_data, batch_size=self.args.batch_size, shuffle=True, drop_last=True,num_workers=0)
            acc = self.eval(test_loader)
            acces.append(acc)
        s = ''
        for i in range(len(acces)):
            s += 'Task ' + str(i) + ': ' + str(acces[i]) + ', '
        logging.info(s)
    def eval(self, valdata):
        self.model.eval()
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(valdata):
            image = image.type(torch.FloatTensor).cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            pred = p[:, :self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        acc = correct / (wrong + correct)
        print("test Acc: {}".format(acc * 100))
        self.initaccs.append(acc * 100)
        self.model.train()
        print("---------------------------------------------")
        return acc

    def save_model_task(self, current_task, model_path):
        torch.save(self.model, model_path + 'Task_' + str(current_task) + '_model.pth')
        print('Task: %d model save scuccess!' % (current_task))

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, model_path,batch_size, epoches, lr):
        self.batch_size = batch_size
        criterion = nn.CrossEntropyLoss()
        dataset = self.dataset
        combined_seen_test=None

        for step_b in range(dataset.batch_num):

            print(f"Incremental step : {step_b }")
            train_data, test_data = dataset.getNextClasses(step_b)

            print(f'number of trainset: {len(train_data)}, number of testset: {len(test_data)}')
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
            test_loader =  DataLoader(test_data , batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

            if step_b == 0:
                combined_seen_test = test_loader.dataset
                old_combined_seen_test = combined_seen_test
            else:
                old_combined_seen_test = combined_seen_test
                combined_seen_test = ConcatDataset([combined_seen_test, test_loader.dataset])

            combined__seen_test_loader = DataLoader(combined_seen_test, batch_size=32, shuffle=True)
            old_combined_seen_test_loader = DataLoader(old_combined_seen_test, batch_size=32, shuffle=True)
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)


            for epoch in range(epoches):

                print("Epoch", epoch)

                self.model.train()
                if step_b >= 1:
                    total_CE_loss,total_kd_loss,total_reg_loss,total_loss_protoAug=\
                        self.stage1_distill(train_loader, criterion, optimizer)
                else:
                    total_loss=self.stage1(train_loader, criterion, optimizer)

                accuracy = self.eval(test_loader)
                print('epoch:%d, accuracy:%.5f' % (epoch, accuracy))
                old_seen_accuracy = self.eval(old_combined_seen_test_loader)
                print('epoch:%d, old_seen_accuracy:%.5f' % (epoch, old_seen_accuracy))
                all_seen_accuracy = self.eval(combined__seen_test_loader)
                print('epoch:%d, all_seen_accuracy:%.5f' % (epoch, all_seen_accuracy))

            self.test(step_b)
            self.save_model_task(step_b,model_path)
            if step_b>=1:
                self.buildPrototype(self.previous_model, train_loader, step_b)

                new_class_previous_model_Prototype = self.prototype[self.seen_cls-self.interval_nums:]
                new_class_previous_model_label =  self.class_label[self.seen_cls-self.interval_nums:]

                self.prototype = self.prototype[:self.seen_cls - self.interval_nums]
                self.class_label = self.class_label[:self.seen_cls - self.interval_nums]

                self.buildPrototype(self.model, train_loader, step_b)

                #重点 原型纠正
                self.correctPrototype(new_class_previous_model_Prototype,new_class_previous_model_label )
            else:
                self.buildPrototype(self.model, train_loader, step_b)

            self.previous_model = copy.deepcopy(self.model)
            self.seen_cls += self.interval_nums

    def prototype_loss_with_separation(self,features, labels, prototypes, available_classes, lambda_sep=0.1):

        intra_class_loss = 0.0
        for i, feature in enumerate(features):
            class_idx = available_classes.tolist().index(labels[i].item())  # Get the index in the available classes
            prototype = prototypes[class_idx]
            intra_class_loss += torch.sum((feature - prototype) ** 2)


        inter_class_loss = 0.0
        num_classes = len(available_classes)
        old_prototypes=torch.tensor(self.prototype).cuda()
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                inter_class_loss += 1.0 / (
                            torch.sum((prototypes[i] - prototypes[j]) ** 2) + 1e-8)  # Avoid division by zero

            for k in range(len(self.prototype)):
                inter_class_loss += 1.0 / (
                            torch.sum((prototypes[i] - old_prototypes[k]) ** 2) + 1e-8)  # Avoid division by zero
        # Total loss
        total_loss = intra_class_loss / len(features) + lambda_sep * inter_class_loss
        return total_loss



    def stage1(self, train_loader, criterion, optimizer):
        print("Training ... ")

        total_loss = 0
        for batch, (imgL, label) in enumerate(train_loader):
            imgL, label = imgL.float().cuda(),  label.long().cuda()
            outs = self.model(imgL)
            loss_cls = criterion(outs[:, :self.seen_cls], label)

            # 梯度反向传播
            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()
            total_loss += loss_cls.item()
        print("loss:{}".format(total_loss))
        self.losses.append(total_loss)
        return total_loss

    def stage1_distill(self, train_loader, criterion, optimizer):
        print("Training ... ")

        T = 2
        beta = (self.seen_cls - 20) / self.seen_cls
        print("classification proportion 1-beta = ", 1 - beta)

        total_CE_loss = 0
        total_kd_loss=0
        total_reg_loss=0
        total_loss_protoAug=0


        for batch, (imgL, label) in enumerate(train_loader):

            imgL,  label = imgL.float().cuda(), label.long().cuda()
            p= self.model(imgL)

            loss_reg = 0

            for (p_param, c_param, layer_lambda) in zip(self.previous_model.parameters(), self.model.parameters(), self.layer_lambdas):
                loss_reg += layer_lambda *(p_param - c_param).norm(2)

            previous_q = self.previous_model(imgL)
            previous_q = F.softmax(previous_q[:, :self.seen_cls - self.interval_nums] / T, dim=1)

            log_current_p = F.log_softmax(p[:, :self.seen_cls - self.interval_nums] / T, dim=1)
            loss_distillation = -torch.mean(torch.sum(previous_q * log_current_p, dim=1))

            loss_crossEntropy = nn.CrossEntropyLoss()(p[:, :self.seen_cls], label)


            isProto = True
            loss_protoAug = 0
            if isProto:
                proto_aug = []
                proto_aug_label = []
                index = list(range(self.seen_cls - self.interval_nums))

                for _ in range(self.batch_size):
                    np.random.shuffle(index)

                    temp = self.prototype[index[0]] + np.random.normal(0, 0.05, 256)  # 512

                    proto_aug.append(temp)
                    proto_aug_label.append(self.class_label[index[0]])

                proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().cuda()
                proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).cuda()

                soft_feat_aug = self.model.fc(proto_aug)
                proto_aug_label = proto_aug_label.to(torch.int64)

                loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug[:, :self.seen_cls]/ 2, proto_aug_label)


            loss = loss_crossEntropy + loss_protoAug +loss_reg



            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


            total_CE_loss += loss_crossEntropy.item()
            total_kd_loss += loss_distillation.item()
            total_reg_loss+= loss_reg.item()
            total_loss_protoAug+=loss_protoAug.item()


        return total_CE_loss,total_kd_loss,total_reg_loss,total_loss_protoAug


    def buildPrototype(self, model, loader,step_b):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(loader):

                feature = model.feature_extract_stage2(images.float().cuda()).reshape(self.batch_size, -1)
                if feature.shape[0] == self.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())


        labels_set = np.unique(labels)  # 0-49
        print(labels_set)

        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])

        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))

        prototype = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))

        if step_b == 0:
            self.prototype = prototype  # [50,512]
            self.class_label = class_label
        else:
            self.prototype = np.concatenate((self.prototype, prototype), axis=0)
            self.class_label = np.concatenate((self.class_label,class_label, ), axis=0)

    def correctPrototype(self,new_class_previous_model_Prototype,new_class_previous_model_label):

        new_Prototypes=self.prototype[self.seen_cls-self.interval_nums:]
        old_Prototypes=self.prototype[:self.seen_cls-self.interval_nums]

        if isinstance(old_Prototypes, np.ndarray):
            old_Prototypes = torch.from_numpy(old_Prototypes)

        if isinstance(new_Prototypes, np.ndarray):
            new_Prototypes = torch.from_numpy(new_Prototypes)

        if isinstance(new_class_previous_model_Prototype, np.ndarray):
            new_class_previous_model_Prototype = torch.from_numpy(new_class_previous_model_Prototype)

        similarity_matrix = torch.mm(old_Prototypes, new_class_previous_model_Prototype.T )  # 计算相似度矩阵
        softmax_similarity_matrix = F.softmax(similarity_matrix, dim=1)

        delta=new_Prototypes-new_class_previous_model_Prototype #偏差，self.model(新类别)-self.previous_model(新类别)

        if isinstance(delta, np.ndarray):
            delta = torch.from_numpy(delta)

        new_delta= torch.mm(softmax_similarity_matrix , delta)  # 计算相似度矩阵
        a=0.97
        old_Prototypes= a*old_Prototypes+(1-a)*new_delta
        self.prototype[:self.seen_cls - self.interval_nums]=old_Prototypes


if __name__ == '__main__':

    init_nums = [50, 50, 50, 40]
    interval_nums = [25, 10, 5, 3]
    total_nc=100
    data_name='ADS-B100'
    for init_num, task_size in zip(init_nums, interval_nums):

        # 总共有多少任务
        task_num = (total_nc - init_num) // task_size
        # 创建或打开日志文件
        log_root = 'log/%s/phases_%d' % (data_name, task_num)

        if not os.path.exists(log_root):
            os.makedirs(log_root)
        logging.basicConfig(filename=log_root + '/Our_training_log.txt', level=logging.INFO)

        model_path = 'log/models/%s/phases_%d/' % (data_name, task_num)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        trianer = Trainer(total_cls=total_nc, init_num_classes=init_num, interval_nums=task_size)
        trianer.train(model_path,batch_size=16, epoches=20, lr=0.01)


