import torch
import numpy as np
import os

def main():

    # Start by defining some dataset details:
    num_subjects       = 40
    num_reps           = 6 # Rep 0 has 0 windows, there is only 6 reps. For some reason, subject 1 has rep 0 elements in the .mats
    num_motions        = 49
    num_channels       = 48
    sampling_frequency = 1000 # This is assumed, check later.
    winsize            = 250 # samples
    wininc             = 100
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
        CNN_data = EMGData(s_train, chosen_class_labels = CNN_classes)











if __name__ == "__main__":
    main()