import torch
import numpy as np
import os
import math
from torch.utils.data import Dataset
from center_loss import CenterLoss

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

        data = self.data[idx]
        labels = self.class_label[idx]

        return data, labels


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
        
        num_windows = math.floor((data.shape[0]-winsize)/wininc)

        st=0
        ed=int(st+winsize * sampling_frequency / 1000)
        for w in range(num_windows):
            training_data.append([subject_id,class_num-1, rep_num,w,data[st:ed,:].transpose()])
            st = int(st+wininc * sampling_frequency / 1000)
            ed = int(ed+wininc * sampling_frequency / 1000)

    np.random.shuffle(training_data)
    np.save("Data/S"+str(subject_id), training_data)

def main():

    # Start by defining some dataset details:
    num_subjects       = 9 # Full dataset has 40, testing with first 9
    num_reps           = 6 # Rep 0 has 0 windows, there is only 6 reps. For some reason, subject 1 has rep 0 elements in the .mats
    num_motions        = 49
    num_channels       = 48
    sampling_frequency = 1000 # This is assumed, check later.
    winsize            = 250
    wininc             = 150
    dataset_characteristics = (num_subjects, num_channels, num_reps, num_motions, winsize, wininc, sampling_frequency)

    channel_shape = [6,8]

    # Data Division parameters (which classes are used to train CNN/AE models)
    ANN_classes = [4,5,12,13] # Hold out classes 5,6,13,14. numbers are zero indexed
    CNN_classes = list(range(0,num_motions))
    for ele in sorted(ANN_classes, reverse=True):
        del CNN_classes[ele]
    CNN_train_reps     = [1 2 3 4]
    CNN_valdation_reps = [5]
    CNN_test_reps      = [6]
    

    # Deep learning parameters
    # CNN parameters
    CNN_batch_size = 32
    CNN_lr         = 0.005
    CNN_num_epochs = 100
    CNN_PLOT_LOSS  = False
    # AE parameters
    AE_batch_size  = 32
    AE_lr          = 0.005
    AE_num_epochs  = 100
    AE_PLOT_LOSS   = False


    # Initialize parameters to be stored
    CNN_accuracy        = np.zeros((num_subjects))
    CNN_training_loss   = np.zeros((num_subjects, CNN_num_epochs))
    CNN_validation_loss = np.zeros((num_subjects, CNN_num_epochs))
    AE_training_loss    = np.zeros((num_subjects, AE_num_epochs))
    AE_validation_loss  = np.zeros((num_subjects, AE_num_epochs))

    for s_train in range(num_subjects):
        
        # If we have the dataset saved already as .npy files, we can load those in
        # Otherwise, make the .npy for that subject.
        if not os.path.exists("Data/S"+str(s_train) + ".npy"):
            make_npy(s_train, dataset_characteristics)
        
        
        CNN_train_data      = EMGData(s_train, chosen_class_labels = CNN_classes, chosen_rep_labels=CNN_train_reps,     channel_shape = channel_shape)
        CNN_validation_data = EMGData(s_train, chosen_class_labels = CNN_classes, chosen_rep_labels=CNN_valdation_reps, channel_shape = channel_shape)


#Conv - 2 conv, 1 flatten, 2 linear, softmax
#centerloss + MSEloss









if __name__ == "__main__":
    main()