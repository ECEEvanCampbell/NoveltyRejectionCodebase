import os
import scipy.io
import numpy as np

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



    for s in range (num_subjects):
        if s < 9:
            subj_path = os.listdir(dataset_location+'/subj0' + str(s + 1))
        else:
            subj_path = os.listdir(dataset_characteristics+'/subj0' + str(s + 1))

        
        for f in subj_path:

            session_id = int(f.split('_')[3][4])
            motion_id  = int(f.split('_')[4])
            rep_id     = int(f.split('_')[5].split('.')[0])
            
            rep_saved = (session_id-1) * num_reps + rep_id

            file_contents = scipy.io.loadmat(dataset_location+'/subj0' + str(s + 1) + "/" + f)
            emg_data = np.transpose(file_contents['emg'])

            if not( os.path.exists(save_location + "/S" + str(s+1)) ):
                os.mkdir(save_location + "/S" + str(s+1))

            np.savetxt(save_location + "/S" + str(s+1) + "/S" + str(s+1) + "_C" + str(motion_id) + "_P1_R" + str(rep_saved) + ".csv", emg_data, delimiter=',')




if __name__ == "__main__":
    main()