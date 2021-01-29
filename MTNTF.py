import torch
import numpy as np
import scipy.io
import torch.utils.data as Data
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class attention_recovery(torch.nn.Module):
    def __init__(self, n_road, n_day, n_time, n_factors,device):
        super(attention_recovery, self).__init__()
        
        self.road_fc = nn.Linear(n_road,n_factors,bias=False)
        self.day_fc =  nn.Linear(n_day,n_factors,bias=False)
        self.time_fc = nn.Linear(n_time,n_factors,bias=False)

        ##########  第一层注意力
        self.att_road_fc1=nn.Sequential(
            nn.Linear(n_factors,n_factors),
            nn.Sigmoid()
        )
        self.att_road_fc2=nn.Sequential(
            nn.Linear(n_factors,n_factors),
            nn.Sigmoid()
        )
        self.att_day_fc1=nn.Sequential(
            nn.Linear(n_factors,n_factors),
            nn.Sigmoid()
        )
        self.att_day_fc2=nn.Sequential(
            nn.Linear(n_factors,n_factors),
            nn.Sigmoid()
        )
        self.att_time_fc1=nn.Sequential(
            nn.Linear(n_factors,n_factors),
            nn.Sigmoid()
        )
        self.att_time_fc2=nn.Sequential(
            nn.Linear(n_factors,n_factors),
            nn.Sigmoid()
        )

        ##########  第二层注意力 
        self.intercat_fc=nn.Sequential(
            nn.Linear(3*n_factors,2*n_factors),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.att_fc1=nn.Sequential(
            nn.Linear(2*n_factors,2*n_factors),
            nn.Sigmoid()
        )
        self.att_fc2=nn.Sequential(
            nn.Linear(2*n_factors,2*n_factors),
            nn.Sigmoid()
        )

        self.intercat_fc1=nn.Sequential(
            nn.Linear(5*n_factors,4*n_factors),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4*n_factors,3*n_factors),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(3*n_factors,2*n_factors),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.intercat_fc2=nn.Sequential(
            nn.Linear(5*n_factors,4*n_factors),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4*n_factors,3*n_factors),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(3*n_factors,2*n_factors),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        ##########  输出层
        self.intercat_fcc=nn.Sequential(
            nn.Linear(2*n_factors,n_factors),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.att_fcc1=nn.Sequential(
            nn.Linear(n_factors,n_factors),
            nn.Sigmoid()
        )
        self.att_fcc2=nn.Sequential(
            nn.Linear(n_factors,n_factors),
            nn.Sigmoid()
        )

        self.intercat_fcc1=nn.Sequential(
            nn.Linear(3*n_factors,2*n_factors),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2*n_factors,n_factors),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.intercat_fcc2=nn.Sequential(
            nn.Linear(3*n_factors,2*n_factors),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2*n_factors,n_factors),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.relu=nn.ReLU()

        self.output_fc1=nn.Linear(n_factors,1)
        self.output_fc2=nn.Linear(n_factors,1)

        self.downsample1=nn.Conv1d(3*n_factors,2*n_factors,1)  
        self.downsample2=nn.Conv1d(2*n_factors,n_factors,1)

        self.downsample3=nn.Conv1d(5*n_factors,2*n_factors,1)
        self.downsample4=nn.Conv1d(5*n_factors,2*n_factors,1)

        self.downsample5=nn.Conv1d(3*n_factors,n_factors,1)
        self.downsample6=nn.Conv1d(3*n_factors,n_factors,1)
        
    def forward(self,road,day,time):
        road_feature=self.road_fc(road)
        day_feature=self.day_fc(day)
        time_feature=self.time_fc(time)
        fusion_feature=torch.cat([road_feature,day_feature,time_feature],dim=1)

        _fusion_feature=self.intercat_fc(fusion_feature)
        fusion_feature=torch.unsqueeze(fusion_feature,2)
        fusion_feature=self.downsample1(fusion_feature)      
        fusion_feature=torch.squeeze(fusion_feature,2)
        _fusion_feature=self.relu(fusion_feature+_fusion_feature)

        att_road_feature1=self.att_road_fc1(road_feature)
        att_road_feature2=self.att_road_fc2(road_feature)
        road_feature1=att_road_feature1*road_feature
        road_feature2=att_road_feature2*road_feature

        att_day_feature1=self.att_day_fc1(day_feature)
        att_day_feature2=self.att_day_fc2(day_feature)
        day_feature1=att_day_feature1*day_feature
        day_feature2=att_day_feature2*day_feature

        att_time_feature1=self.att_time_fc1(time_feature)
        att_time_feature2=self.att_time_fc2(time_feature)
        time_feature1=att_time_feature1*time_feature
        time_feature2=att_time_feature2*time_feature

        att_fc1=self.att_fc1(_fusion_feature)
        att_fc2=self.att_fc2(_fusion_feature)
        fc1=att_fc1*_fusion_feature
        fc2=att_fc2*_fusion_feature

        fusion_feature1=torch.cat([road_feature1,day_feature1,time_feature1,fc1],dim=1)
        fusion_feature2=torch.cat([road_feature2,day_feature2,time_feature2,fc2],dim=1)

        _fusion_feature1=self.intercat_fc1(fusion_feature1)
        _fusion_feature2=self.intercat_fc2(fusion_feature2)

        fusion_feature1=torch.unsqueeze(fusion_feature1,2)
        fusion_feature1=self.downsample3(fusion_feature1)  
        fusion_feature1=torch.squeeze(fusion_feature1,2)
        fusion_feature2=torch.unsqueeze(fusion_feature2,2)
        fusion_feature2=self.downsample4(fusion_feature2)  
        fusion_feature2=torch.squeeze(fusion_feature2,2)

        _fusion_feature1=self.relu(_fusion_feature1+fusion_feature1)
        _fusion_feature2=self.relu(_fusion_feature2+fusion_feature2)

        __fusion_feature=self.intercat_fcc(_fusion_feature)
        _fusion_feature=torch.unsqueeze(_fusion_feature,2)
        _fusion_feature=self.downsample2(_fusion_feature)  
        _fusion_feature=torch.squeeze(_fusion_feature,2)
        __fusion_feature=self.relu(_fusion_feature+__fusion_feature)

        att_fcc1=self.att_fcc1(__fusion_feature)
        att_fcc2=self.att_fcc2(__fusion_feature)
        fcc1=att_fcc1*__fusion_feature
        fcc2=att_fcc2*__fusion_feature

        _fusion_feature1_=torch.cat([_fusion_feature1,fcc1],dim=1)
        _fusion_feature2_=torch.cat([_fusion_feature2,fcc2],dim=1)

        __fusion_feature1=self.intercat_fcc1(_fusion_feature1_)
        __fusion_feature2=self.intercat_fcc2(_fusion_feature2_)

        _fusion_feature1_=torch.unsqueeze(_fusion_feature1_,2)
        _fusion_feature1_=self.downsample5(_fusion_feature1_) 
        _fusion_feature1_=torch.squeeze(_fusion_feature1_,2)
        _fusion_feature2_=torch.unsqueeze(_fusion_feature2_,2)
        _fusion_feature2_=self.downsample6(_fusion_feature2_)  
        _fusion_feature2_=torch.squeeze(_fusion_feature2_,2)

        __fusion_feature1=self.relu(_fusion_feature1_+__fusion_feature1)
        __fusion_feature2=self.relu(_fusion_feature2_+__fusion_feature2)

        output1=self.output_fc1(__fusion_feature1)
        output2=self.output_fc2(__fusion_feature2)

        return output1,output2

class balance_net(torch.nn.Module):
  def __init__(self, n_road, n_day, n_time, n_factors,loss,device):
    super(balance_net, self).__init__()
  
    self.attention_recovery=attention_recovery(n_road, n_day, n_time, n_factors,device)
    self.sigma1=nn.Parameter(data=torch.ones(1,1),requires_grad=True).to(device)
    self.sigma2=nn.Parameter(data=torch.ones(1,1),requires_grad=True).to(device)

    self.loss_function=loss

  def forward(self,road,day,time,item1,item2):
    prediction1,prediction2=self.attention_recovery(road,day,time)

    item1=item1.view(prediction1.shape)
    item2=item2.view(prediction2.shape)
    pos1=torch.where(item1!=0)
    pos2=torch.where(item2!=0)

    loss1=self.loss_function(prediction1[pos1],item1[pos1])
    loss2=self.loss_function(prediction2[pos2],item2[pos2])

    total_loss=1/(2*self.sigma1*self.sigma1)*loss1+1/(2*self.sigma2*self.sigma2)*loss2+torch.log(self.sigma1*self.sigma2)

    return self.sigma1.item(),self.sigma2.item(),prediction1,prediction2,total_loss

def train_attention_recovery(dense_tensor1,random_tensor1,missing_rate1,dense_tensor2,random_tensor2,missing_rate2,n_factors=20):
    n_road=dense_tensor1.shape[0]
    n_day=dense_tensor1.shape[1]
    n_time=dense_tensor1.shape[2]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    binary_tensor1=np.round(random_tensor1+0.5-missing_rate1)
    # binary_tensor1 = np.zeros(dense_tensor1.shape)
    # for i1 in range(dense_tensor1.shape[0]):
    #   for i2 in range(dense_tensor1.shape[1]):
    #     binary_tensor1[i1, i2, :] = np.round(random_tensor1[i1, i2] + 0.5 - missing_rate1)
    sparse_tensor1=np.multiply(dense_tensor1,binary_tensor1)

    binary_tensor2=np.round(random_tensor2+0.5-missing_rate2)
    # binary_tensor2 = np.zeros(dense_tensor2.shape)
    # for i1 in range(dense_tensor2.shape[0]):
    #   for i2 in range(dense_tensor2.shape[1]):
    #     binary_tensor2[i1, i2, :] = np.round(random_tensor2[i1, i2] + 0.5 - missing_rate2)
    sparse_tensor2=np.multiply(dense_tensor2,binary_tensor2)

    train_pos1=np.where(sparse_tensor1!=0)
    train_pos2=np.where(sparse_tensor2!=0)

    test_pos1=np.where((dense_tensor1!=0) & (sparse_tensor1==0))
    test_pos2=np.where((dense_tensor2!=0) & (sparse_tensor2==0))

    max_value1=np.max(sparse_tensor1[train_pos1])
    min_value1=np.min(sparse_tensor1[train_pos1])
    sparse_tensor1[train_pos1]=(sparse_tensor1[train_pos1]-min_value1)/(max_value1-min_value1)

    max_value2=np.max(sparse_tensor2[train_pos2])
    min_value2=np.min(sparse_tensor2[train_pos2])
    sparse_tensor2[train_pos2]=(sparse_tensor2[train_pos2]-min_value2)/(max_value2-min_value2)

    train_data=[]
    test_data=[]

    for i in range(dense_tensor1.shape[0]):
        for j in range(dense_tensor1.shape[1]):
            for k in range(dense_tensor1.shape[2]):
                road,day,time=i,j,k
                item1,item2=sparse_tensor1[i,j,k],sparse_tensor2[i,j,k]
                train_data.append([road,day,time,item1,item2])

    for i in range(dense_tensor2.shape[0]):
        for j in range(dense_tensor2.shape[1]):
            for k in range(dense_tensor2.shape[2]):
                road,day,time=i,j,k
                test_data.append([road,day,time])

    train_data=torch.from_numpy(np.array(train_data)).to(device)
    test_data=torch.from_numpy(np.array(test_data)).to(device)

    batch_size,lr,num_epochs=256,0.001,1000
    train_iter=Data.DataLoader(train_data,batch_size,shuffle=True)
    loss = torch.nn.MSELoss()
    net=balance_net(n_road, n_day, n_time, n_factors,loss,device)    ## 
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    test_loss_rmse1=np.zeros((num_epochs,1))
    test_loss_mae1=np.zeros((num_epochs,1))
    test_loss_mape1=np.zeros((num_epochs,1))
    test_loss_rmse2=np.zeros((num_epochs,1))
    test_loss_mae2=np.zeros((num_epochs,1))
    test_loss_mape2=np.zeros((num_epochs,1))
    train_loss1=np.zeros((num_epochs,1))
    train_loss2=np.zeros((num_epochs,1))
  
    for epoch in range(num_epochs):
        train_loss=0.0
        for data in train_iter:
            road,day,time=data[:,0].long(),data[:,1].long(),data[:,2].long()

            road_onehot=F.one_hot(road,num_classes=n_road)
            day_onehot=F.one_hot(day,num_classes=n_day)
            time_onehot=F.one_hot(time,num_classes=n_time)

            item1,item2=data[:,3].float(),data[:,4].float()

            sigma1,sigma2,prediction1,prediction2,total_loss=net(road_onehot.float(),day_onehot.float(),time_onehot.float(),item1,item2)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss+=total_loss.item()

        net.eval()
        road,day,time=test_data[:,0].long(),test_data[:,1].long(),test_data[:,2].long()
        road_onehot=F.one_hot(road,num_classes=n_road)
        day_onehot=F.one_hot(day,num_classes=n_day)
        time_onehot=F.one_hot(time,num_classes=n_time)

        sigma1,sigma2,prediction1,prediction2,total_loss=net(road_onehot.float(),day_onehot.float(),time_onehot.float(),torch.Tensor(dense_tensor1),torch.Tensor(dense_tensor2))
        prediction1=prediction1.view(n_road,n_day,n_time)
        prediction2=prediction2.view(n_road,n_day,n_time)

        prediction1=prediction1.cpu().data.numpy()
        prediction2=prediction2.cpu().data.numpy()

        prediction1=prediction1*(max_value1-min_value1)+min_value1
        prediction2=prediction2*(max_value2-min_value2)+min_value2

        rmse1=np.sum((dense_tensor1[test_pos1]-prediction1[test_pos1])*(dense_tensor1[test_pos1]-prediction1[test_pos1]))/dense_tensor1[test_pos1].shape[0]
        mae1=np.sum(np.abs(dense_tensor1[test_pos1]-prediction1[test_pos1]))/dense_tensor1[test_pos1].shape[0]
        mape1=np.sum(np.abs((dense_tensor1[test_pos1]-prediction1[test_pos1])/dense_tensor1[test_pos1]))/dense_tensor1[test_pos1].shape[0]*100

        rmse2=np.sum((dense_tensor2[test_pos2]-prediction2[test_pos2])*(dense_tensor2[test_pos2]-prediction2[test_pos2]))/dense_tensor2[test_pos2].shape[0]
        mae2=np.sum(np.abs(dense_tensor2[test_pos2]-prediction2[test_pos2]))/dense_tensor2[test_pos2].shape[0]
        mape2=np.sum(np.abs((dense_tensor2[test_pos2]-prediction2[test_pos2])/dense_tensor2[test_pos2]))/dense_tensor2[test_pos2].shape[0]*100
        

        test_loss_rmse1[epoch,0]=np.sqrt(rmse1)
        test_loss_rmse2[epoch,0]=np.sqrt(rmse2)
        test_loss_mae1[epoch,0]=mae1
        test_loss_mae2[epoch,0]=mae2
        test_loss_mape1[epoch,0]=mape1
        test_loss_mape2[epoch,0]=mape2

        print('epoch: {}, test_loss_rmse1: {},test_loss_mae1:{},test_loss_mape1:{},test_loss_rmse2:{},test_loss_mae2:{},test_loss_mape2:{},sigma1:{},sigma2:{}'.format(
      epoch + 1, np.sqrt(rmse1),mae1,mape1,np.sqrt(rmse2),mae2,mape2,sigma1,sigma2))
    final_rmse1=np.mean(test_loss_rmse1[-10:,0])
    final_mae1=np.mean(test_loss_mae1[-10:,0])
    final_mape1=np.mean(test_loss_mape1[-10:,0])
    final_rmse2=np.mean(test_loss_rmse2[-10:,0])
    final_mae2=np.mean(test_loss_mae2[-10:,0])
    final_mape2=np.mean(test_loss_mape2[-10:,0])
    print(final_rmse1,final_mae1,final_mape1,final_rmse2,final_mae2,final_mape2)

    return test_loss_rmse1,test_loss_mae1,test_loss_mape1,test_loss_rmse2,test_loss_mae2,test_loss_mape2,final_rmse1,final_mae1,final_mape1,final_rmse2,final_mae2,final_mape2 


dense_tensor1=np.load('shanghai_speed_tensor.npy')
random_tensor1=np.load('pointwise_missing_speed_tensor.npy')

dense_tensor2=np.load('shanghai_volume_tensor.npy')
random_tensor2=np.load('pointwise_missing_volume_tensor.npy')

missing_rate1,missing_rate2=0.1,0.1
test_loss_rmse1,test_loss_mae1,test_loss_mape1,test_loss_rmse2,test_loss_mae2,test_loss_mape2,final_rmse1,final_mae1,final_mape1,final_rmse2,final_mae2,final_mape2 =train_attention_recovery(dense_tensor1,random_tensor1,missing_rate1,dense_tensor2,random_tensor2,missing_rate2)