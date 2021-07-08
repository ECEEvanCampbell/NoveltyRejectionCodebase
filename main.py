import torch
import numpy as np
import os
import math


def compute_center_loss(features, centers, targets):
    features = features.view(features.size(0), -1)
    target_centers = centers[targets]
    criterion = torch.nn.MSELoss()
    center_loss = criterion(features, target_centers)
    return center_loss

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

    # Data Division parameters (which classes are used to train CNN/AE models)
    CNN_classes = []
    ANN_classes = []

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
        
        # TODO: make this class
        #CNN_data = EMGData(s_train, chosen_class_labels = CNN_classes, channel_shape = [6,8], features = ['MAV','RMS','WL'])


#Conv - 2 conv, 1 flatten, 2 linear, softmax
#centerloss + MSEloss









if __name__ == "__main__":
    main()