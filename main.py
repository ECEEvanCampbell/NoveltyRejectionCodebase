# Basic Modules
import numpy as np
import os
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
    def __init__(self, subject_number, chosen_rep_labels=None, chosen_class_labels=None, channel_shape = [6,8]):

        if isinstance(subject_number, list):
            data = []
            for n in subject_number:
                data.append(np.load("Data/S{}.npy".format(str(n)), allow_pickle=True))

            subject_data = np.concatenate(data)

        else:
            subject_data = np.load("Data/S{}.npy".format(str(subject_number)), allow_pickle=True)

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
        
        features = extract_sEMG_features(data)
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

def train(CNN_model, CNN_train_loader, CNN_classes, optimizer, device, alpha=0.00005):
    # Train the recognition model
    # Model.train - enable gradient tracking, enable batch normalization, dropout
    CNN_model.train()
    # store losses of this epoch in a list (element = loss on batch)
    losses_class = []
    losses_group = []

    num_classes = len(CNN_classes)
    center_loss = CenterLoss(num_classes=num_classes, feat_dim=CNN_model.fc2.in_features, use_gpu=True)

    for batch_idx, (data, label) in enumerate(CNN_train_loader):
        for label_idx, (l) in enumerate(label):
            label[label_idx] = CNN_classes.index(l)
        # Send data, labels to GPU if available
        data = data.to(device)
        label = label.to(device)
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


def validate(CNN_model, CNN_validation_loader, CNN_classes, device, alpha=0.00005):
    # Evaluate the recognition model
    # Model.eval - disable gradient tracking, enable batch normalization, dropout
    CNN_model.eval()
    # store losses of this epoch in a list (element = loss on batch)
    losses_class = []
    losses_group = []
    num_classes = len(CNN_classes)
    center_loss = CenterLoss(num_classes=num_classes, feat_dim=CNN_model.fc2.in_features, use_gpu=True)

    for batch_idx, (data, label) in enumerate(CNN_validation_loader):
        for label_idx, (l) in enumerate(label):
            label[label_idx] = CNN_classes.index(l)
        # Send data, labels to GPU if available
        data = data.to(device)
        label = label.to(device)
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

def test(CNN_model, CNN_test_loader, CNN_classes, device):
    # Evaluate the model
    # model.eval - disable gradient tracking, batch normalization, dropout
    CNN_model.eval()
    # Keep track of correct samples
    correct = 0

    for batch_idx, (data, label) in enumerate(CNN_test_loader):
        # Send data, labels to GPU if GPU is available
        for label_idx, (l) in enumerate(label):
            label[label_idx] = CNN_classes.index(l)
        data = data.to(device)
        label = label.to(device)
        # Passing data to model calls the forward method.
        _, output_class = CNN_model(data)
        predictions = output_class.argmax(dim=-1)
        # Add up correct samples from batch
        for i, prediction in enumerate(predictions):
            correct += int(prediction == label[i])
    # Return average accuracy 
    return float(correct/ len(CNN_test_loader.dataset))

def get_CNN_features(CNN_model, CNN_loader, device):
    CNN_model.eval()
    output_features = torch.empty((CNN_loader.dataset.features.shape[0], CNN_model.fc2.in_features))
    labels = torch.empty((CNN_loader.dataset.features.shape[0]))
    for batch_idx, (data, label) in enumerate(CNN_loader):
        data = data.to(device)
        output_features[batch_idx*CNN_loader.batch_size:(batch_idx+1)*CNN_loader.batch_size,:], _ = CNN_model(data)
        labels[batch_idx*CNN_loader.batch_size:(batch_idx+1)*CNN_loader.batch_size] = label

    return output_features, labels


def build_data_loader(batch_size, num_workers, pin_memory, data):
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle = True,
        collate_fn = collate_fn,
        num_workers = num_workers,
        pin_memory = pin_memory
    )
    return data_loader

def collate_fn(batch):
    # Populate these lists from the batch
    signals = torch.empty((len(batch),batch[0][0].shape[0], batch[0][0].shape[1],batch[0][0].shape[2] ))
    labels = torch.empty((len(batch)))
    for idx, (signal, label) in enumerate(batch):
        signals[idx,:,:,:] = signal
        labels[idx] = label
    return signals, labels.long()



def extract_sEMG_features(data):
    
    features = np.zeros((data.shape[0], data.shape[1]*3), dtype=float)
    if torch.is_tensor(data):
        data = data.numpy()
    features[:,0:data.shape[1]] = getRMSfeat(data)
    features[:,data.shape[1]:2*data.shape[1]] = getMAVfeat(data)
    features[:,2*data.shape[1]:3*data.shape[1]] = getWLfeat(data)

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


def make_npy(subject_id, dataset_characteristics, base_dir="Data/Raw_Data"):
    # Inside of dataset folder, get list of all files associated with the subject of subject_id
    (num_subjects, num_channels, num_reps, num_motions, winsize, wininc, sampling_frequency) = dataset_characteristics
    subj_path = os.listdir(base_dir+'/S' + str(subject_id + 1))
    training_data = []
    # For this list:
    for f in subj_path:
        # Get the identifiers in the filename
        path = os.path.join(base_dir,"S"+ str(subject_id+1),f)
        class_num = int(f.split('_')[1][1:])
        rep_num   = int(f.split('_')[3][1])

        # load the file
        data = np.genfromtxt(path,delimiter=',')
        b, a = signal.iirnotch( 60/ (sampling_frequency/2), 20)
        notched_data = signal.lfilter(b,a, data,axis=0)
        b, a = signal.butter(N=4, Wn=[20/(sampling_frequency/2), 450/(sampling_frequency/2)],btype="band")
        filtered_data = signal.lfilter(b,a, notched_data,axis=0)
        num_windows = math.floor((filtered_data.shape[0]-winsize)/wininc)

        st=0
        ed=int(st+winsize * sampling_frequency / 1000)
        for w in range(num_windows):
            training_data.append([subject_id,class_num-1, rep_num,w,filtered_data[st:ed,:].transpose()])
            st = int(st+wininc * sampling_frequency / 1000)
            ed = int(ed+wininc * sampling_frequency / 1000)

    np.random.shuffle(training_data)
    np.save("Data/S"+str(subject_id), training_data)

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

    # Start by defining some dataset details:
    num_subjects       = 10 # Full dataset has 40, testing with first 10
    num_reps           = 6 # Rep 0 has 0 windows, there is only 6 reps. For some reason, subject 1 has rep 0 elements in the .mats
    num_motions        = 49 # The full number is 49, but my PC can't handle that many
    num_channels       = 12
    sampling_frequency = 1000 # This is assumed, check later.
    winsize            = 250
    wininc             = 150
    dataset_characteristics = (num_subjects, num_channels, num_reps, num_motions, winsize, wininc, sampling_frequency)

    channel_shape = [3,4]

    # Data Division parameters (which classes are used to train CNN/AE models)
    ANN_classes = [4,5,12,13] # Hold out classes 5,6,13,14. numbers are zero indexed
    CNN_classes = list(range(0,num_motions))
    for ele in sorted(ANN_classes, reverse=True):
        del CNN_classes[ele]
    CNN_train_reps     = [1, 2, 3, 4]
    CNN_valdation_reps = [5]
    CNN_test_reps      = [6]
    

    # Deep learning parameters
    # CNN parameters
    CNN_batch_size = 32
    CNN_lr         = 0.005
    CNN_weight_decay = 0.001
    CNN_num_epochs = 100
    CNN_PLOT_LOSS  = True
    # AE parameters
    AE_batch_size  = 32
    AE_lr          = 0.005
    AE_num_epochs  = 100
    AE_PLOT_LOSS   = False


    # Initialize parameters to be stored
    CNN_accuracy        = np.zeros((num_subjects))
    CNN_train_class_loss      = np.zeros((num_subjects, CNN_num_epochs))
    CNN_validation_class_loss = np.zeros((num_subjects, CNN_num_epochs))
    CNN_train_group_loss      = np.zeros((num_subjects, CNN_num_epochs))
    CNN_validation_group_loss = np.zeros((num_subjects, CNN_num_epochs))
    AE_train_loss       = np.zeros((num_subjects, AE_num_epochs))
    AE_validation_loss  = np.zeros((num_subjects, AE_num_epochs))

    for s_train in range(num_subjects):
        
        # If we have the dataset saved already as .npy files, we can load those in
        # Otherwise, make the .npy for that subject.
        if not os.path.exists("Data/S"+str(s_train) + ".npy"):
            make_npy(s_train, dataset_characteristics)
        
        # BEGIN THE CNN TRAINING PROCEDURE
        # Get the datasets prepared for training and testing
        CNN_train_data      = EMGData(s_train, chosen_class_labels = CNN_classes, chosen_rep_labels=CNN_train_reps,     channel_shape = channel_shape)
        CNN_validation_data = EMGData(s_train, chosen_class_labels = CNN_classes, chosen_rep_labels=CNN_valdation_reps, channel_shape = channel_shape)
        # Define the dataloaders that prepare batches of data
        CNN_train_loader      = build_data_loader(CNN_batch_size, num_workers, pin_memory, CNN_train_data) 
        CNN_validation_loader = build_data_loader(CNN_batch_size, num_workers, pin_memory, CNN_validation_data) 
        # Instantiate the CNN model.
        CNN_model = CNNModel(n_output=len(CNN_classes),i_depth=3, nch=num_channels) # 3 refers to initial depth of input (RMS, WL, MAV Maps)
        CNN_model.to(device)
        # Training setup:
        optimizer = optim.Adam(CNN_model.parameters(), lr=CNN_lr, weight_decay = CNN_weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=0.02, patience=3, factor=0.2)

        for epoch in range(0, CNN_num_epochs):
            CNN_train_class_loss[s_train,epoch], CNN_train_group_loss[s_train, epoch]         = train(   CNN_model, CNN_train_loader,  CNN_classes,   optimizer, device)
            CNN_validation_class_loss[s_train,epoch], CNN_validation_group_loss[s_train,epoch] = validate(CNN_model, CNN_validation_loader, CNN_classes,          device)

            scheduler.step(CNN_validation_class_loss[s_train, epoch])

        CNN_test_data     = EMGData(s_train, chosen_class_labels = CNN_classes, chosen_rep_labels=CNN_test_reps,     channel_shape = channel_shape)
        CNN_test_loader   = build_data_loader(CNN_batch_size, num_workers, pin_memory, CNN_test_data) 
        CNN_accuracy[s_train] = test(CNN_model, CNN_test_loader, CNN_classes, device)

        if CNN_PLOT_LOSS:
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
            axs[1].legend()

            output_features, labels= get_CNN_features(CNN_model, CNN_test_loader, device)
            tsne = TSNE(n_components=2)
            projected_data = tsne.fit_transform(output_features.detach().numpy())

            for class_num in CNN_classes:
                class_ids = labels == class_num
                axs[2].scatter(projected_data[class_ids,0],projected_data[class_ids,1],label=str(class_num))
            axs[2].set(xlabel="tsne1",ylabel="tsne2")
            axs[2].set_title("TSNE")
            plt.show()

            

if __name__ == "__main__":
    main()