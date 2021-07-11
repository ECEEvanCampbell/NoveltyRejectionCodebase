

% Get list of .mat files

files = dir([pwd '/S*.mat']);

for f = 1:length(files)
    
    load([files(f).folder '/' files(f).name],...
        'emg','acc','rerepetition','restimulus');
    
    class_list = unique(restimulus);
    rep_list = unique(rerepetition);
    
    tmp = split(files(f).name,'_');
    subj = str2double(tmp{1}(2:end));
    clear tmp
    
    
    for cc = 2:length(class_list)
        % class_list always has 0 (rest class)
        % This rest class is ~ half the total samples. 
        % For now do not include (start at id 2)
        class = class_list(cc);
        
        for rr = rr_span
            
            rep = rep_list(rr);
            cl_re_id = and(restimulus == class_list(cc), rerepetition == rep_list(rr));
            
            % EMG, ACC, and info Feature Extraction
            data = [double(emg(cl_re_id,:))];% We don't want ACC for this study double(acc(cl_re_id,:))];
            
            if ~exist(['S' num2str(subj)],'dir')
                mkdir(['S' num2str(subj)])
            end
            csvwrite([pwd '/S' num2str(subj) '/S' num2str(subj) '_C' num2str(class) '_P' num2str(1) '_R' num2str(rep) '.csv'],data);
            
            
            
        end
    end
end