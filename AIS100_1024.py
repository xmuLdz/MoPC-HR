import torch
import numpy as np


device = "cuda:0" if torch.cuda.is_available() else "cpu"


class AIS100_1024:
    def __init__(self, total_classes,init_nums, interval_nums):

        #ADS-B100
        # E:\pycharm_code\PyCI_AIS\src\ADS107\data
        # self.train_data = np.load(r"E:\pycharm_code\PyCI_AIS\src\ADS107\data\ADS_train_data.npy")   #.reshape(-1,3072)
        # self.train_labels = np.load(r"E:\pycharm_code\PyCI_AIS\src\ADS107\data\ADS_train_label.npy")
        # self.test_data = np.load(r"E:\pycharm_code\PyCI_AIS\src\ADS107\data\ADS_test_data.npy")      #.reshape(-1,3072)
        # self.test_labels = np.load(r"E:\pycharm_code\PyCI_AIS\src\ADS107\data\ADS_test_label.npy")

        #只有90类的ADS-B数据
        # self.train_data = np.load(r"ADS_train_data.npy")
        # self.train_labels = np.load(r"ADS_train_label.npy")
        # self.test_data = np.load(r"ADS_test_data.npy")
        # self.test_labels = np.load(r"ADS_test_label.npy")

        # 只有10类的USRP数据
        # self.train_data = np.load(r"E:\pycharm_code\work2_linux\Few_FedL\dataset\Usrp_Gmsk\Gmsk_train_data.npy")
        # self.train_labels = np.load(r"E:\pycharm_code\work2_linux\Few_FedL\dataset\Usrp_Gmsk\Gmsk_train_label.npy")
        # self.test_data = np.load(r"E:\pycharm_code\work2_linux\Few_FedL\dataset\Usrp_Gmsk\Gmsk_test_data.npy")
        # self.test_labels = np.load(r"E:\pycharm_code\work2_linux\Few_FedL\dataset\Usrp_Gmsk\Gmsk_test_label.npy")

        #AIS100-100
        self.train_data = np.load(r"E:\pycharm_code\PyCI_AIS\src\AIS100_WA\data\train_data_steady_1024_6_4.npy")
        self.train_labels = np.load(r"E:\pycharm_code\PyCI_AIS\src\AIS100_WA\data\train_label_steady_1024_6_4.npy")
        self.test_data = np.load(r"E:\pycharm_code\PyCI_AIS\src\AIS100_WA\data\test_data_steady_1024_6_4.npy")
        self.test_labels = np.load(r"E:\pycharm_code\PyCI_AIS\src\AIS100_WA\data\test_label_steady_1024_6_4.npy")

        self.init_nums = init_nums
        self.interval_nums = interval_nums
        self.total_classes = total_classes
        self.batch_num = (self.total_classes - init_nums) // interval_nums + 1  # 间隔
        self.train_groups, self.test_groups,self.all_test,self.stage_test100 = self.initialize()

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def standardization(self, data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma

    def initialize(self):
        train_test_low = self.init_nums
        interval_nums = self.interval_nums
        train_groups = [[] for _ in range(self.batch_num)]
        test_groups = [[] for _ in range(self.batch_num)]
        stage_test100=[[] for _ in range(self.batch_num)]
        all_test=[[]]
        for i in range(100):
            index = np.where(i == self.train_labels)[0]
            temp_data = self.train_data[index[:200]]
            temp_labels = self.train_labels[index[:200]]
            if i==0:
                train_data_=temp_data
                train_labels=temp_labels
            else:
                train_data_=np.concatenate((train_data_,temp_data),axis=0)
                train_labels = np.concatenate((train_labels, temp_labels), axis=0)

        for i in range(len(train_groups)):

            for train_data, train_label in zip(train_data_, train_labels):

                train_data = self.normalization(train_data)


                if i == 0:
                    if train_label < train_test_low:
                        train_groups[i].append((train_data, train_label))
                elif  train_test_low-interval_nums <= train_label < train_test_low :

                    train_groups[i].append((train_data, train_label))
            ########################测试集组装#############################################
            for test_data, test_label in zip(self.test_data, self.test_labels):
                test_data = self.normalization(test_data)


                if i == 0 :
                    all_test[0].append((test_data, test_label))
                    if test_label < train_test_low:
                        test_groups[i].append((test_data, test_label))
                        stage_test100[i].append((test_data, test_label))
                elif train_test_low-interval_nums <= test_label < train_test_low :
                    test_groups[i].append((test_data, test_label))
                    stage_test100[i].append((test_data, test_label))
            if i>0:
                stage_test100[i]=stage_test100[i-1]+stage_test100[i] #拼接前一个任务的数据集

            train_test_low = train_test_low + interval_nums
        return train_groups, test_groups,all_test,stage_test100

    def getNextClasses(self, step_b):
        return self.train_groups[step_b], self.test_groups[step_b]

    def getStageTestClasses(self, step_b):
        return self.stage_test100[step_b]

    def getAllTest(self):
        return self.all_test[0]


if __name__ == '__main__':
    pass



