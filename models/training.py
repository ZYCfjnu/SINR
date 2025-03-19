from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import time
from models.utils import nearest_distances, self_nearest_distances_K

class Trainer(object):
    def __init__(self, model, device, train_dataset, val_dataset, exp_path, fine_tuning=True, optimizer='Adam', lr = 1e-4, threshold = 0.1):
        self.model = model.to(device)
        self.device = device
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr= lr)
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters())
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.fineTuning = fine_tuning
        self.exp_path = exp_path
        if not os.path.exists(self.exp_path):
            print(self.exp_path)
            os.makedirs(self.exp_path)
        self.writer = SummaryWriter(self.exp_path + 'summary/')
        self.val_min = None
        self.max_dist = threshold
        self.criterionL1Loss = torch.nn.L1Loss(reduction='none').to(self.device)
        self.sampled_num = 40000

    def train_step(self,batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_loss(self, batch):
        device = self.device
        input_voxel = batch.get('inputs').to(device)
        points = batch.get('points').float().to(device)
        B, _, _, _ = input_voxel.shape
        sampled_points = torch.rand(B, self.sampled_num, 3).float().to(self.device)
        sampled_points = sampled_points * 2. - 1.
        df_pred = self.model(sampled_points, input_voxel)
        df_pred = torch.reshape(df_pred, (B * self.sampled_num, -1))
        gt_nearest_distances = nearest_distances(sampled_points.transpose(2, 1), points.transpose(2, 1))
        points_distances = self_nearest_distances_K(points.transpose(2, 1), k=3)
        k_points_distances = points_distances.topk(k=20, dim=1)[0]
        points_distance = torch.mean(k_points_distances, dim=1)
        points_distance = points_distance.repeat(1, self.sampled_num).reshape(-1, self.sampled_num, 1)
        gt_nearest_distances -= points_distance / 2.
        gt_nearest_distances = torch.clamp(gt_nearest_distances, min=0)
        gt_nearest_distances = torch.reshape(gt_nearest_distances, (B * self.sampled_num, -1))
        if self.max_dist > 0:
            loss_rec = self.criterionL1Loss(torch.clamp(df_pred, max=self.max_dist), torch.clamp(gt_nearest_distances, max=self.max_dist))
        else:
            loss_rec = self.criterionL1Loss(df_pred, gt_nearest_distances)
        loss = loss_rec.mean()
        return loss

    def train_model(self, epochs):
        loss = 0
        train_data_loader = self.train_dataset.get_loader()
        start, training_time = self.load_checkpoint()
        iteration_start_time = time.time()
        for epoch in range(start, epochs):
            sum_loss = 0
            print('Start epoch {}'.format(epoch))
            for batch in train_data_loader:
                iteration_duration = time.time() - iteration_start_time
                if iteration_duration > 500:
                    training_time += iteration_duration
                    iteration_start_time = time.time()
                    val_loss = self.compute_val_loss()
                    if self.val_min is None:
                        self.val_min = val_loss
                    if val_loss < self.val_min:
                        self.val_min = val_loss
                        for path in glob(self.exp_path + 'val_min=*'):
                            os.remove(path)
                        np.save(self.exp_path + 'val_min={}'.format(epoch), [epoch, val_loss])
                    self.writer.add_scalar('val loss batch avg', val_loss, epoch)
                    print("Current val_loss: {}".format(val_loss))
                loss = self.train_step(batch)
                print("Current train_loss: {}".format(loss))
                sum_loss += loss
            self.writer.add_scalar('training loss last batch', loss, epoch)
            self.writer.add_scalar('training loss batch avg', sum_loss / len(train_data_loader), epoch)
            self.save_checkpoint(epoch, training_time)


    def save_checkpoint(self, epoch, training_time):
        path = self.exp_path + 'checkpoints/checkpoint_{}.tar'.format(epoch)
        if not os.path.exists(path):
            torch.save({ 'training_time': training_time ,
                         'epoch':epoch,
                         'model_state_dict': self.model.state_dict(),
                         'optimizer_state_dict': self.optimizer.state_dict()},
                         path)

    def load_checkpoint(self):
        checkpoints = glob(self.exp_path+'checkpoints/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.exp_path))
            return 0,0
        else:
            checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=int)
            checkpoints = np.sort(checkpoints)
            path = self.exp_path + 'checkpoints/checkpoint_{}.tar'.format(checkpoints[-1])
            print('Loaded checkpoint from: {}'.format(path))
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            training_time = checkpoint['training_time']
            return epoch, training_time

    def compute_val_loss(self):
        self.model.eval()
        sum_val_loss = 0
        num_batches = 15
        for _ in range(num_batches):
            try:
                val_batch = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_dataset.get_loader().__iter__()
                val_batch = self.val_data_iterator.next()
            sum_val_loss += self.compute_loss(val_batch).item()
        return sum_val_loss / num_batches

def convertMillis(millis):
    seconds = int((millis / 1000) % 60)
    minutes = int((millis / (1000 * 60)) % 60)
    hours = int((millis / (1000 * 60 * 60)))
    return hours, minutes, seconds

def convertSecs(sec):
    seconds = int(sec % 60)
    minutes = int((sec / 60) % 60)
    hours = int((sec / (60 * 60)))
    return hours, minutes, seconds