# Basic Modules
import numpy as np
import os
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import math
import random
from scipy import signal
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# Deep Learning Modules
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Custom Modules from others -- author: KaiyangZhou
from center_loss import CenterLoss



class CNNModel(nn.Module):
    def __init__(self, n_output, i_depth=3, nch=12):
        super().__init__()
        # How many filters we want per layer
        filters = 32
        linear_nodes = 512
        input_to_linear = nch*filters


        # What layers do we want
        self.conv1 = nn.Conv2d(i_depth, filters, kernel_size=(3,3), padding=1)
        self.bn1   = nn.BatchNorm2d(filters)

        self.conv2 = nn.Conv2d(filters, filters, kernel_size=(3,3), padding=1)
        self.bn2   = nn.BatchNorm2d(filters)


        self.fc1 = nn.Linear(input_to_linear, linear_nodes)
        self.fc2 = nn.Linear(linear_nodes, n_output)

        self.drop = nn.Dropout(p=0.2)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Forward pass: input x, output probabilities of predicted class
        # First layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.drop(x)

        # Second Layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.drop(x)


        # Convert to linear layer suitable input
        x = x.view(x.shape[0],-1)
       
        # First linear layer, leave x for centerloss
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
         # final layer: linear layer that outputs N_Class neurons
        y = self.fc2(x)
        y = F.softmax(y, dim=1)

        return x,y

class AEModel(nn.Module):
    def __init__(self, input_nodes):
        super().__init__()

        self.encode_fc1 = nn.Linear(input_nodes, 256)
        self.encode_fc2 = nn.Linear(256, 64)

        self.decode_fc1 = nn.Linear(64,256)
        self.decode_fc2 = nn.Linear(256, input_nodes)

        self.activation = nn.ReLU()
        

    def forward(self, x):
        x = self.encode_fc1(x)
        x = self.activation(x)
        x = self.encode_fc2(x)
        x = self.activation(x)

        xhat = self.decode_fc1(x)
        xhat = self.activation(xhat)
        
        xhat = self.decode_fc2(xhat)
        xhat = self.activation(xhat)

        return xhat

def fix_random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

class EMGData(Dataset):
    def __init__(self, subject_number, dataset, chosen_rep_labels=None, chosen_class_labels=None, channel_shape = [6,8],  buffer_channels = 0):

        if isinstance(subject_number, list):
            data = []
            for n in subject_number:
                data.append(np.load("Data/" + dataset + "_S{}.npy".format(str(n)), allow_pickle=True))

            subject_data = np.concatenate(data)

        else:
            subject_data = np.load("Data/" + dataset + "_S{}.npy".format(str(subject_number)), allow_pickle=True)

        if chosen_rep_labels is not None:
            subject_data = [i for i in subject_data if i[2] in chosen_rep_labels]

        if chosen_class_labels is not None:
            subject_data = [i for i in subject_data if i[1] in chosen_class_labels]
            
        self.subject_number = subject_number

        # extract classes
        self.class_label = torch.tensor([i[1] for i in subject_data], dtype=torch.float)
        self.rep_label   = torch.tensor([i[2] for i in subject_data], dtype=torch.int)
        self.num_labels = torch.unique(self.class_label).shape[0]

        data = torch.tensor([i[4] for i in subject_data], dtype=torch.float)
        
        features = extract_sEMG_features(data, buffer_channels)
        features = features.reshape((features.shape[0], 3, channel_shape[0], channel_shape[1]))
        self.features = torch.tensor(features)

    def __len__(self):
        return len(self.class_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.features[idx]
        labels = self.class_label[idx]

        return data, labels

def CNN_train(CNN_model, CNN_train_loader, closedset_classes, optimizer, device, alpha=0.00005, transform=None):
    # Train the recognition model
    # Model.train - enable gradient tracking, enable batch normalization, dropout
    CNN_model.train()
    # store losses of this epoch in a list (element = loss on batch)
    losses_class = []
    losses_group = []

    num_classes = len(closedset_classes)
    center_loss = CenterLoss(num_classes=num_classes, feat_dim=CNN_model.fc2.in_features, use_gpu=True)

    for batch_idx, (data, label) in enumerate(CNN_train_loader):
        for label_idx, (l) in enumerate(label):
            label[label_idx] = closedset_classes.index(l)
        # Send data, labels to GPU if available
        data = data.to(device)
        label = label.to(device)

        if transform is not None:
            data = transform(data)
            data = data.to(device).to(torch.float)

        # Passing data to model calls the forward method.
        output_features, output_class = CNN_model(data)
        # Output: (batch_size, 1, n_class)
        loss_class = F.cross_entropy(output_class, label)
        loss_group = center_loss(output_features, label)
        total_loss = loss_class + alpha * loss_group
        # reset optimizer buffer
        optimizer.zero_grad()
        # send loss backwards
        total_loss.backward()
        # Update weights
        optimizer.step()
        # store the losses of this batch
        losses_class.append(loss_class.item())
        losses_group.append(loss_group.item())
    return sum(losses_class)/len(losses_class),  sum(losses_group)/len(losses_group)

def AE_train(AE_model, AE_train_loader, optimizer, device):
    # Train the feature reconstruction model
    # Model.train - enable gradient tracking, enable batch normalization, dropout
    AE_model.train()
    # store losses of this epoch in a list (element = loss on batch)
    losses = []

    for batch_idx, (data) in enumerate(AE_train_loader):
        # Send data, labels to GPU if available
        data = data.to(device)
        # Passing data to model calls the forward method.
        output_features = AE_model(data)
        # Output: (batch_size, n_features)
        loss = F.l1_loss(output_features, data)
        # reset optimizer buffer
        optimizer.zero_grad()
        # send loss backwards
        loss.backward(retain_graph=True)
        # Update weights
        optimizer.step()
        # store the losses of this batch
        losses.append(loss.item())
    return sum(losses)/len(losses)

def CNN_validate(CNN_model, CNN_validation_loader, closedset_classes, device, alpha=0.00005, transform=None):
    # Evaluate the recognition model
    # Model.eval - disable gradient tracking, enable batch normalization, dropout
    CNN_model.eval()
    # store losses of this epoch in a list (element = loss on batch)
    losses_class = []
    losses_group = []
    num_classes = len(closedset_classes)
    center_loss = CenterLoss(num_classes=num_classes, feat_dim=CNN_model.fc2.in_features, use_gpu=True)

    for batch_idx, (data, label) in enumerate(CNN_validation_loader):
        for label_idx, (l) in enumerate(label):
            label[label_idx] = closedset_classes.index(l)
        # Send data, labels to GPU if available
        data = data.to(device)
        label = label.to(device)
        if transform is not None:
            data = transform(data)
            data = data.to(device).to(torch.float)
        # Passing data to model calls the forward method.
        output_features, output_class = CNN_model(data)
        # Output: (batch_size, 1, n_class)
        loss_class = F.cross_entropy(output_class, label)
        loss_group = center_loss(output_features, label)

        # No optimizer stuff to be done

        # store the losses of this batch
        losses_class.append(loss_class.item())
        losses_group.append(loss_group.item())
    return sum(losses_class)/len(losses_class),  sum(losses_group)/len(losses_group)

def AE_validate(AE_model, AE_validation_loader, device):
    # Validate the feature reconstruction model
    # Model.eval - disable gradient tracking, enable batch normalization, dropout
    AE_model.eval()
    # store losses of this epoch in a list (element = loss on batch)
    losses = []

    for batch_idx, (data) in enumerate(AE_validation_loader):
        # Send data, labels to GPU if available
        data = data.to(device)
        # Passing data to model calls the forward method.
        output_features = AE_model(data)
        # Output: (batch_size, n_features)
        loss = F.l1_loss(output_features, data)
        
        # No optimizer stuff to be done

        # store the losses of this batch
        losses.append(loss.item())
    return sum(losses)/len(losses)

def CNN_test(CNN_model, CNN_test_loader, closedset_classes, device, transform=None):
    # Evaluate the model
    # model.eval - disable gradient tracking, batch normalization, dropout
    CNN_model.eval()
    # Keep track of correct samples
    correct = 0

    for batch_idx, (data, label) in enumerate(CNN_test_loader):
        # Send data, labels to GPU if GPU is available
        for label_idx, (l) in enumerate(label):
            label[label_idx] = closedset_classes.index(l)
        data = data.to(device)
        label = label.to(device)
        if transform is not None:
            data = transform(data)
            data = data.to(device).to(torch.float)
        # Passing data to model calls the forward method.
        _, output_class = CNN_model(data)
        predictions = output_class.argmax(dim=-1)
        # Add up correct samples from batch
        for i, prediction in enumerate(predictions):
            correct += int(prediction == label[i])
    # Return average accuracy 
    return float(correct/ len(CNN_test_loader.dataset))

def cascade_test(CNN_model, AE_model, CNN_test_loader, AE_test_loader, closedset_classes, rejection_threshold, device):
    CNN_model.eval()
    AE_model.eval()
    correct   = 0
    rejection = 0
    AE_data_iterator = iter(AE_test_loader)
    predictions = np.zeros((CNN_test_loader.dataset.class_label.shape[0]))
    
    for batch_idx, (CNN_data, label) in enumerate(CNN_test_loader):

        for label_idx, (l) in enumerate(label):
            label[label_idx] = closedset_classes.index(l)

        AE_data = next(AE_data_iterator)
        AE_data = AE_data.to(device)
        CNN_data = CNN_data.to(device)
        label = label.to(device)

        _, output_class = CNN_model(CNN_data)
        predictions = output_class.argmax(dim=-1)

        AE_reconstruction = AE_model(AE_data)
        AE_loss = F.l1_loss(AE_reconstruction, AE_data, reduction='none').mean(axis=1)

        for i, prediction in enumerate(predictions):
            # Check if we reject
            if AE_loss[i] < rejection_threshold:
                # If not, check if it is correct
                correct += int(prediction == label[i])
            else:
                rejection += 1
    # CNN accuracy rejection = correct samples / non rejected samples
    CNN_accuracy_rejection = correct / (CNN_test_loader.dataset.class_label.shape[0] - rejection)
    # false rejection rate = rejected samples / total samples
    false_rejection_rate = rejection / (CNN_test_loader.dataset.class_label.shape[0])
    
    return CNN_accuracy_rejection, false_rejection_rate

def get_CNN_features(CNN_model, CNN_loader, device):
    CNN_model.eval()
    output_features = torch.empty((CNN_loader.dataset.features.shape[0], CNN_model.fc2.in_features))
    labels = torch.empty((CNN_loader.dataset.features.shape[0]))
    for batch_idx, (data, label) in enumerate(CNN_loader):
        data = data.to(device)
        output_features[batch_idx*CNN_loader.batch_size:(batch_idx+1)*CNN_loader.batch_size,:], _ = CNN_model(data)
        labels[batch_idx*CNN_loader.batch_size:(batch_idx+1)*CNN_loader.batch_size] = label

    return output_features, labels

def build_CNN_data_loader(batch_size, num_workers, pin_memory, data):
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle = True,
        collate_fn = CNN_collate_fn,
        num_workers = num_workers,
        pin_memory = pin_memory
    )
    return data_loader

def build_AE_data_loader(batch_size, num_workers, pin_memory, data):
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle = True,
        collate_fn = AE_collate_fn,
        num_workers = num_workers,
        pin_memory = pin_memory
    )
    return data_loader

def CNN_collate_fn(batch):
    # Populate these lists from the batch
    signals = torch.empty((len(batch),batch[0][0].shape[0], batch[0][0].shape[1],batch[0][0].shape[2] ))
    labels = torch.empty((len(batch)))
    for idx, (sample, label) in enumerate(batch):
        signals[idx,:,:,:] = sample
        labels[idx] = label
    return signals, labels.long()

def AE_collate_fn(batch):
    signals = torch.empty((len(batch),batch[0].shape[0]))
    for idx, (sample) in enumerate(batch):
        signals[idx,:] = sample
    return signals

def extract_sEMG_features(data, buffer_channels = 0):
    
    features = np.zeros((data.shape[0], 3, (data.shape[1]+buffer_channels)), dtype=float)
    if torch.is_tensor(data):
        data = data.numpy()
    features[:,0, 0:data.shape[1]] = getRMSfeat(data)
    features[:,1, 0:data.shape[1]] = getMAVfeat(data)
    features[:,2, 0:data.shape[1]] = getWLfeat(data)

    return features

def getMAVfeat(signal):
    feat = np.mean(np.abs(signal),2)
    return feat

def getRMSfeat(signal):
    feat = np.sqrt(np.mean(np.square(signal),2))
    return feat

def getWLfeat(signal):
    feat = np.sum(np.abs(np.diff(signal,axis=2)),2)
    return feat

def make_npy(subject_id, dataset, dataset_characteristics, base_dir="Data/Raw_Data"):
    # Inside of dataset folder, get list of all files associated with the subject of subject_id
    (num_subjects, num_channels, buffer_channels, num_reps, num_motions, winsize, wininc, sampling_frequency) = dataset_characteristics
    subj_path = os.listdir(base_dir+'/' + dataset + '/S' + str(subject_id + 1))
    training_data = []
    # For this list:
    for f in subj_path:
        # Get the identifiers in the filename
        path = os.path.join(base_dir,dataset,"S"+ str(subject_id+1),f)
        class_num = int(f.split('_')[1][1:])
        rep_part  = f.split('_')[3]
        rep_num   = int(rep_part.split('.')[0][1:])

        # load the file
        data = np.genfromtxt(path,delimiter=',')
        b, a = signal.iirnotch( 60/ (sampling_frequency/2), 20)
        notched_data = signal.lfilter(b,a, data,axis=0)
        b, a = signal.butter(N=4, Wn=[20/(sampling_frequency/2), 450/(sampling_frequency/2)],btype="band")
        filtered_data = signal.lfilter(b,a, notched_data,axis=0)
        num_windows = math.floor((data.shape[0]-winsize* round(sampling_frequency / 1000))/(wininc* round(sampling_frequency / 1000)))

        st=0
        ed=int(st+winsize * round(sampling_frequency / 1000))
        for w in range(num_windows):
            training_data.append([subject_id,class_num-1, rep_num,w,data[st:ed,:].transpose()])
            st = int(st+wininc * round(sampling_frequency / 1000))
            ed = int(ed+wininc * round(sampling_frequency / 1000))

    np.random.shuffle(training_data)
    np.save("Data/" + dataset + "_S"+str(subject_id), training_data)

def get_AE_rejection_threshold(AE_model, AE_validation_loader, device, rejection_tolerance=1.5):
    # Validate the feature reconstruction model
    # Model.eval - disable gradient tracking, enable batch normalization, dropout
    AE_model.eval()
    # store losses of this epoch in a list (element = loss on batch)
    losses = np.zeros((AE_validation_loader.dataset.shape[0]))

    for batch_idx, (data) in enumerate(AE_validation_loader):
        # Send data, labels to GPU if available
        data = data.to(device)
        # Passing data to model calls the forward method.
        output_features = AE_model(data)
        # Output: (batch_size, 1, n_class)
        loss = F.l1_loss(output_features, data, reduction='none').mean(axis=1)
        
        # No optimizer stuff to be done

        # store the losses of this batch
        losses[batch_idx*AE_validation_loader.batch_size:(1+batch_idx)*AE_validation_loader.batch_size] = loss.detach().cpu().numpy()
    
    mean_loss = losses.mean()
    std_loss = losses.std()
    
    rejection_threshold = mean_loss + rejection_tolerance*std_loss
    return rejection_threshold, losses

def outlier_test(AE_model, AE_outlier_loader, AE_rejection_threshold, device):
    AE_model.eval()
    rejection = 0

    for batch_idx, data in enumerate(AE_outlier_loader):

        data = data.to(device)
        
        AE_reconstruction = AE_model(data)
        AE_loss = F.l1_loss(AE_reconstruction, data, reduction='none').mean(axis=1)

        for i, loss in enumerate(AE_loss):
            # Check if we reject
            if AE_loss[i] > AE_rejection_threshold:
                rejection += 1
        
    # positive rejection rate = rejected samples / total samples
    positive_rejection_rate = rejection / (AE_outlier_loader.dataset.shape[0])

    return positive_rejection_rate

class Normalize(object):
    def __init__(self, data, device):

        min, _ = torch.min(data, 0) # Metric across training samples
        for i in range(len(min.shape)-1):
            min, _ = torch.min(min, -1)   # Metric across feature channels (RMS, etc.)
        self.min = min.to(device)

        max, _ = torch.max(data, 0)
        for i in range(len(max.shape)-1):
            max, _ = torch.max(max, -1)   # Metric across feature channels (RMS, etc.)
        self.max = max.to(device)

        d = self._normalize(data.to(device)) 

        self.mean = torch.mean(d, dim=(0, 2, 3)).to(device)
        self.std = torch.std(d, dim=(0, 2, 3)).to(device)

    def __call__(self, x):
        x = self._normalize(x)
        x = x - self.mean[None, :, None, None]
        x = x / self.std[None, :, None, None]
        return x

    def _normalize(self, x):
        return (x - self.min[None, :, None, None]) / (self.max[None, :, None, None] - self.min[None, :, None, None])

def main():

    # Fix the random seed -- make results reproducible
    # Found in utils.py, this sets the seed for the random, torch, and numpy libraries.
    fix_random_seed(1, torch.cuda.is_available())
    # get device available for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # My PC sucks, so I can't run things in parallel. If yours can, uncomment these lines
        #num_workers = 2
        #pin_memory  = True
        num_workers = 0
        pin_memory = False
    else:
        num_workers = 0
        pin_memory  = False

    dataset = "SEEDS"#"Ninapro2"
    winsize = 250
    wininc = 150

    if dataset == "Ninapro2":
        # Start by defining some dataset details:
        num_subjects       = 40 # Full dataset has 40, testing with first 10
        num_reps           = 6 # Rep 0 has 0 windows, there is only 6 reps. For some reason, subject 1 has rep 0 elements in the .mats
        num_motions        = 49 
        num_channels       = 12
        sampling_frequency = 1000 # This is assumed, check later.
        channel_shape      = [3,4]
        buffer_channels    = 0
        outlier_classes    = [5,6,13,14] # Hold out classes 5,6,13,14. numbers are zero indexed
        
    elif dataset == "SEEDS":
        num_subjects       = 5 # Dataset has 25 subjects, but use 5 as pilot data
        num_reps           = 18
        num_motions        = 13
        num_channels       = 134
        sampling_frequency = 2048
        buffer_channels    = 2 # zero channels to make rectangular input.
        channel_shape      = [8, 17]
        outlier_classes    = [1,2]

    dataset_characteristics = (num_subjects, num_channels, buffer_channels, num_reps, num_motions, winsize, wininc, sampling_frequency)

    train_reps = list(range(1,num_reps+1))
    test_reps = [train_reps.pop(-1)]
    validation_reps = [train_reps.pop(-1)]
    closedset_classes  = list(range(0,num_motions))
    for ele in sorted(outlier_classes, reverse=True):
        del closedset_classes[ele]

    

    # Data Division parameters (which classes are used to train CNN/AE models)
    
    

    # Deep learning parameters
    # CNN parameters
    CNN_batch_size = 32
    CNN_lr         = 0.005
    CNN_weight_decay = 0.001
    CNN_num_epochs = 100
    CNN_PLOT_LOSS  = True
    # AE parameters
    AE_batch_size  = 32
    AE_lr          = 0.01
    AE_weight_decay = 0.001
    AE_num_epochs  = 5
    AE_PLOT_LOSS   = True
    PLOT_REJECTION = True


    # Initialize parameters to be stored
    CNN_accuracy              = np.zeros((num_subjects))
    CNN_train_class_loss      = np.zeros((num_subjects, CNN_num_epochs))
    CNN_validation_class_loss = np.zeros((num_subjects, CNN_num_epochs))
    CNN_train_group_loss      = np.zeros((num_subjects, CNN_num_epochs))
    CNN_validation_group_loss = np.zeros((num_subjects, CNN_num_epochs))
    AE_train_loss             = np.zeros((num_subjects, AE_num_epochs))
    AE_validation_loss        = np.zeros((num_subjects, AE_num_epochs))
    CNN_accuracy_rejection    = np.zeros((num_subjects))
    false_rejection_rate      = np.zeros((num_subjects))
    positive_rejection_rate   = np.zeros((num_subjects))

    for s_train in range(num_subjects):
        
        # If we have the dataset saved already as .npy files, we can load those in
        # Otherwise, make the .npy for that subject.
        if not os.path.exists("Data/" + dataset +  "_S"+str(s_train) + ".npy"):
            make_npy(s_train, dataset, dataset_characteristics)
        
        # BEGIN THE CNN TRAINING PROCEDURE
        # Get the datasets prepared for training and testing
        # Procedure 1: prepare the data in feature image format

        CNN_train_data      = EMGData(s_train, dataset, chosen_class_labels = closedset_classes, chosen_rep_labels=train_reps,     channel_shape = channel_shape, buffer_channels = buffer_channels)
        norm_transform = Normalize(CNN_train_data.features.data, device)
        CNN_train_loader      = build_CNN_data_loader(CNN_batch_size, num_workers, pin_memory, CNN_train_data) 
        del CNN_train_data

        CNN_validation_data = EMGData(s_train, dataset, chosen_class_labels = closedset_classes, chosen_rep_labels=validation_reps, channel_shape = channel_shape, buffer_channels = buffer_channels)
        CNN_validation_loader = build_CNN_data_loader(CNN_batch_size, num_workers, pin_memory, CNN_validation_data) 
        del CNN_validation_data
        
        # Procedure 2: train a CNN model using the closed set gestures.
        # Instantiate the CNN model.
        CNN_model = CNNModel(n_output=len(closedset_classes),i_depth=3, nch=num_channels+buffer_channels) # 3 refers to initial depth of input (RMS, WL, MAV Maps)
        
        # Training setup:

        if os.path.exists(f"Models/{dataset}_S{s_train}.cnn"):
            CNN_model.load_state_dict(torch.load(f"Models/{dataset}_S{s_train}.cnn"))
            CNN_model.to(device)
        else:
            CNN_model.to(device)
            CNN_optimizer = optim.Adam(CNN_model.parameters(), lr=CNN_lr, weight_decay = CNN_weight_decay)
            CNN_scheduler = optim.lr_scheduler.ReduceLROnPlateau(CNN_optimizer, 'min', threshold=0.02, patience=3, factor=0.2)

            for epoch in range(0, CNN_num_epochs):
                CNN_train_class_loss[s_train,epoch], CNN_train_group_loss[s_train, epoch]          = CNN_train(   CNN_model, CNN_train_loader,  closedset_classes,   CNN_optimizer, device, transform=norm_transform)
                CNN_validation_class_loss[s_train,epoch], CNN_validation_group_loss[s_train,epoch] = CNN_validate(CNN_model, CNN_validation_loader, closedset_classes,          device    , transform=norm_transform)

                CNN_scheduler.step(CNN_validation_class_loss[s_train, epoch])

            torch.save(CNN_model.state_dict(), f"Models/{dataset}_S{s_train}.cnn")

        CNN_test_data     = EMGData(s_train, dataset,chosen_class_labels = closedset_classes, chosen_rep_labels=test_reps,     channel_shape = channel_shape, buffer_channels=buffer_channels)
        CNN_test_loader   = build_CNN_data_loader(CNN_batch_size, num_workers, pin_memory, CNN_test_data)
        del CNN_test_data
        CNN_accuracy[s_train] = CNN_test(CNN_model, CNN_test_loader, closedset_classes, device, transform=norm_transform)

        if CNN_PLOT_LOSS:
            # CNN plot loss flag should only be enabled when the training loop is performed (no model is saved)
            fig, axs = plt.subplots(3)
            fig.suptitle('CNN Loss Analysis')
            axs[0].plot(CNN_train_class_loss[s_train,:], label="train_class_loss")
            axs[0].plot(CNN_validation_class_loss[s_train,:], label="validation_class_loss")
            axs[0].set(xlabel="Epoch",ylabel="Loss")
            axs[0].set_title('Class Loss (Cross Entropy)')
            axs[0].legend()

            axs[1].plot(CNN_train_group_loss[s_train,:], label="train_group_loss")
            axs[1].plot(CNN_validation_group_loss[s_train,:], label="validation_group_loss")
            axs[1].set(xlabel="Epoch",ylabel="Loss")
            axs[1].set_title('Center Loss (Cross Entropy)')
            axs[1].legend()

            output_features, labels= get_CNN_features(CNN_model, CNN_test_loader, device)
            tsne = TSNE(n_components=2)
            projected_data = tsne.fit_transform(output_features.detach().numpy())

            for class_num in closedset_classes:
                class_ids = labels == class_num
                axs[2].scatter(projected_data[class_ids,0],projected_data[class_ids,1],label=str(class_num))
            axs[2].set(xlabel="tsne1",ylabel="tsne2")
            axs[2].set_title("TSNE")
            fig.savefig(f"Figures/{dataset}_S{s_train}_ClosedSetCNNTraining.png")
            plt.clf()

        # Procedure 3: Reject novel samples using AE.
        AE_train_data, _      = get_CNN_features(CNN_model, CNN_train_loader, device)
        AE_train_loader      = build_AE_data_loader(AE_batch_size, num_workers, pin_memory, AE_train_data)
        del AE_train_data

        AE_validation_data, _ = get_CNN_features(CNN_model, CNN_validation_loader, device)
        AE_validation_loader = build_AE_data_loader(AE_batch_size, num_workers, pin_memory, AE_validation_data)
        del AE_validation_data

        AE_test_data, _       = get_CNN_features(CNN_model, CNN_test_loader, device)
        AE_test_loader       = build_AE_data_loader(AE_batch_size, num_workers, pin_memory, AE_test_data)
        del AE_test_data

        # Get the features and build loader for them.

        AE_model = AEModel(input_nodes = CNN_model.fc2.in_features)

        if os.path.exists(f"Models/{dataset}_S{s_train}.ae"):
            AE_model.load_state_dict(torch.load(f"Models/{dataset}_S{s_train}.ae"))
            AE_model.to(device)
        else:
            AE_model.to(device)


            AE_optimizer = optim.Adam(AE_model.parameters(), lr=AE_lr, weight_decay = AE_weight_decay)#optim.SGD(AE_model.parameters(),lr=AE_lr)
            AE_scheduler = optim.lr_scheduler.ReduceLROnPlateau(AE_optimizer, 'min', threshold=0.02, patience=3, factor=0.2)

            for epoch in range(0, AE_num_epochs):
                AE_train_loss[s_train,epoch]      = AE_train(   AE_model, AE_train_loader,   AE_optimizer, device)
                AE_validation_loss[s_train,epoch] = AE_validate(AE_model, AE_validation_loader,            device)

                AE_scheduler.step(AE_validation_loss[s_train, epoch])

            torch.save(AE_model.state_dict(), f"Models/{dataset}_S{s_train}.ae")

            if AE_PLOT_LOSS:
                
                plt.plot(AE_train_loss[s_train,:], label="train_loss")
                plt.plot(AE_validation_loss[s_train,:], label="validation_loss")
                plt.xlabel(xlabel="Epoch")
                plt.ylabel(ylabel="Loss")
                plt.title(label='Class Loss (Cross Entropy)')
                plt.legend()
                plt.savefig(f"Figures/{dataset}_S{s_train}_AELoss.png")
                plt.clf()


        # Get the threshold for rejection with trained AE model
        AE_rejection_threshold, validation_losses = get_AE_rejection_threshold(AE_model, AE_validation_loader, device)
        if PLOT_REJECTION:

            CNN_outlier_data        = EMGData(s_train, dataset, chosen_class_labels = outlier_classes, chosen_rep_labels=None, channel_shape = channel_shape, buffer_channels=buffer_channels)
            CNN_outlier_loader      = build_CNN_data_loader(CNN_batch_size, num_workers, pin_memory, CNN_outlier_data) 
            AE_outlier_data, _      = get_CNN_features(CNN_model, CNN_outlier_loader, device)
            AE_outlier_loader       = build_AE_data_loader(AE_batch_size, num_workers, pin_memory, AE_outlier_data)
            _, validation_losses_outlier = get_AE_rejection_threshold(AE_model, AE_outlier_loader, device)

            plt.hist(validation_losses,100,alpha=0.5,color='b')
            plt.hist(validation_losses_outlier,100,alpha=0.5,color='r')
            plt.axvline(x=AE_rejection_threshold)
            plt.xlabel(xlabel="Loss")
            plt.ylabel(ylabel="Frequency of Occurance")
            plt.title(label="Rejection Threshold from AE Validation Loss")
            plt.savefig(f"Figures/{dataset}_S{s_train}_RejectionThreshold.png")
            plt.clf()

        # Get final metrics (4):

        # 1: We have closed set gesture accuracy already from the CNN_accuracy variable

        # 2: Does closed set accuracy go up when using rejection (do we reject more incorrect samples than correct?)
        # 3: Get the "false rejection rate": closed set samples that were deemed outliers.
        CNN_accuracy_rejection[s_train], false_rejection_rate[s_train] = cascade_test(CNN_model, AE_model, CNN_test_loader, AE_test_loader, closedset_classes, AE_rejection_threshold, device)
        
        # false_rejection_rate
        # 4: Get the "positive rejection rate": unknown class samples that were not rejected.
        # positive_rejection_rate
        
        CNN_outlier_data        = EMGData(s_train, dataset, chosen_class_labels = outlier_classes, chosen_rep_labels=None, channel_shape = channel_shape, buffer_channels=buffer_channels)
        CNN_outlier_loader      = build_CNN_data_loader(CNN_batch_size, num_workers, pin_memory, CNN_outlier_data) 
        AE_outlier_data, _      = get_CNN_features(CNN_model, CNN_outlier_loader, device)
        AE_outlier_loader       = build_AE_data_loader(AE_batch_size, num_workers, pin_memory, AE_outlier_data)

        positive_rejection_rate[s_train] = outlier_test(AE_model, AE_outlier_loader, AE_rejection_threshold, device)

        print(f"{dataset}_Subject {s_train}:",
            f"Closed Set Accuracy: {round(100*CNN_accuracy[s_train])} ",
            f"Closed Set Accuracy w/ Rejection: {round(100*CNN_accuracy_rejection[s_train])} ",
            f"False Rejection Rate: {round(100*false_rejection_rate[s_train])} ",
            f"Positive Rejection Rate: {round(100*positive_rejection_rate[s_train])}",
            sep="\n \t")

if __name__ == "__main__":
    main()