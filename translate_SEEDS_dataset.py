import os
import scipy

def main():
    #This points to the base location of the 25 subject folders 
    dataset_location = "D:\Data\[EMG-KIN] SEED"
    # This points to where the csv files will be saved
    save_location    = "Data/Raw_Data/SEEDS"

    print("Converting .mat dataset to .csv")

    num_subjects = 25
    num_reps     = 6
    num_motions  = 13
    num_sessions = 3
    sampling_frequency = 2048
    dataset_characteristics = (num_subjects, num_reps, num_motions, num_sessions, sampling_frequency)





if __name__ == "__main__":
    main()