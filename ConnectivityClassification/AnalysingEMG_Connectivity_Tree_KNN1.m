%EMG data analysis and classification (https://uk.mathworks.com/help/signal/ug/classify-arm-motions-using-emg-signals-and-deep-learning.html)
%Include arfit folder, covm, mvar and the connectivity processing function
clear all;clc;
fs = 5120;
%create datastore type with all the data
datasetFolder = fullfile("EMGData");

SignalDatastore = signalDatastore(datasetFolder,IncludeSubFolders=true,SampleRate=fs); 
p = endsWith(SignalDatastore.Files,"d.mat");
EMGsignal = subset(SignalDatastore,p); 

%include and associate labels
LabelsDatastore = signalDatastore(datasetFolder,SignalVariableNames=["motion";"data_start";"data_end"],IncludeSubfolders=true);  
p = endsWith(LabelsDatastore.Files,"i.mat");
EMGLabel = subset(LabelsDatastore,p);  

%define the region of interest for each motion based on the labels. We remove the first label value (-1) since it is asociated with rest time. The other labels are converted into an array

labels = {};

i = 1;
while hasdata(EMGLabel)

    label = read(EMGLabel);
    
    idx_start = label{2}(2:end)';
    idx_end =label{3}(2:end)';
    value = categorical(label{1}(2:end)',[0 1 2 3 4 6 7 8 9], ...
          [ "Relax" "Idle" "Fist" "Flexion" "Extension" "PinchIndex" "PinchMiddle" "PinchRing" "PinchSmall"]);
    ROI = [idx_start idx_end];
 
    LabelTable = table(ROI,value);
    labels{i} = {LabelTable};

    i = i+1;
end


% New datastore containing the modified label data and display the ROI table from the first observation
LabelDatastore = signalDatastore(labels); 
LabelsTable = preview(LabelDatastore);
LabelsTable{1}

% Combine the signal and label data into one datastore.
DataStore = combine(EMGsignal,LabelDatastore); 

%Preview combine data of first observation (subject 1, session 1)
combinedData = preview(DataStore)

%% Create a signal mask and call plotsigroi to display the labeled motion regions for the first channel of the first signal (only used for visualization of signals)
% figure
% msk = signalMask(combinedData{2});
% plotsigroi(msk,combinedData{1}(:,1))
% 
% %represent the signal after filtering/dowsampling
% 
% % Filter and downsample signal data
% sig=combinedData{1};
% roiTable = combinedData{2};
% 
% % Remove first and last rest periods from signal
% sig(roiTable.ROI(end,2):end,:) = [];
% sig(1:roiTable.ROI(1,1),:) = [];
% 
% % Shift ROI limits to account for deleting start and end of signal
% roiTable.ROI = roiTable.ROI-(roiTable.ROI(1,1)-1);
% 
% % Create signal mask
% m = signalMask(roiTable);
% L = length(sig);
% 
% 
% % Obtain sequence of category labels and remove relax and idle motions
% mask = catmask(m,L);
% 
% sigfilt = lowpass(sig,50,fs);
% downsig = downsample(sigfilt,3);
% 
% % Downsample label data
% downmask = downsample(mask,3);
% 
% idx = ~ismember(downmask,{'Relax', 'Idle'}); 
% 
% downmask = downmask(idx);
% downsig = downsig(idx,:);
% 
% % Create new signal mask without relax and idle categories
% m2 = signalMask(downmask);
% 
% m2.SpecifySelectedCategories = true;
% m2.SelectedCategories = [3 4 5 6 7 8 9];
% downmask = catmask(m2);
% 
% %Downsample frequency
% f=fs/3;
% fs_downsample=ceil(f);
% 
% targetLength = length(downsig); %no chuncks
% % Get number of chunks
% numChunks = 1;
% 
% % Truncate signal and mask to integer number of chunks
% sig = downsig(1:numChunks*targetLength,:);
% mask = downmask(1:numChunks*targetLength);
% 
% % Create a cell array containing signal chunks
% 
% lblOut = reshape(mask,targetLength,1)';
% lblOut = num2cell(lblOut,2);
% 
% figure(2)
% msk = signalMask(lblOut{1});
% plotsigroi(msk,sig(:,1))
% 

%% preprocess data and apply connectivity for each segment of data (1s segment)
trainDatastore = transform(DataStore,@ConnectivityProcessing_v2); 
transformedData = preview(trainDatastore)

%after downsample, the fs is changed. Downsampled by factor of 3 for consistency
f=fs/3;
fs_downsample=ceil(f);



%% load data
traindata = readall(trainDatastore,"UseParallel",true);


%% Convert Training  data from cells of arrays and categorical data to table containing the predictors and observations and categorical containing features.
%and call the classification learner to compute the classification using
%several models.

B=cell2mat(traindata(:,1)');
ans=B';
X= array2table(ans); % observations x predictors (channels) table

Y=grp2idx(traindata{1,2});
for i=2:size(traindata(:,2),1)
    n1=grp2idx(traindata{i,2}); %n1 contains the features as non-catagorical data (numbers from 1 to 7 representing each category) for each cell array
    Y=cat(1,Y,n1); %Y contains the array observations x 1, with the list of features as doubles. 
end

classificationLearner
