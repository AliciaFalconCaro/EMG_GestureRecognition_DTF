function Tsds = EMGProcessingAndConnectivityAnalysis(inputDS)

% uses slidding windows, overlap and average of frequency. The labels are
% averaged for each slidding window.

% Filter and downsample signal data
sig = inputDS{1};
roiTable = inputDS{2};
fs=5120;

% Remove first and last rest periods from signal
sig(roiTable.ROI(end,2):end,:) = [];
sig(1:roiTable.ROI(1,1),:) = [];

% Shift ROI limits to account for deleting start and end of signal
roiTable.ROI = roiTable.ROI-(roiTable.ROI(1,1)-1);

% Create signal mask
m = signalMask(roiTable);
L = length(sig);


% Obtain sequence of category labels and remove relax and idle motions
mask = catmask(m,L);

sigfilt = lowpass(sig,100,fs);
downsig = downsample(sigfilt,3);

% Downsample label data
downmask = downsample(mask,3);

idx = ~ismember(downmask,{'Relax', 'Idle'}); 

downmask = downmask(idx);
downsig = downsig(idx,:);

% Create new signal mask without relax and idle categories
m2 = signalMask(downmask);

m2.SpecifySelectedCategories = true;
m2.SelectedCategories = [3 4 5 6 7 8 9];
downmask = catmask(m2);

%Downsample frequency
f=fs/3;
fs_downsample=ceil(f);



targetLength = floor(fs_downsample/2); %0.5s Window length
overlap = fix(targetLength/4); % Overlap 25%
start_point = 1;
NumberChunks = 1;
Mode1 = 10; % ARFIT(10), Multichannel Yule-Walker(1) ---> BioSig toolbox 

y=downsig;
CH = size(y,2); % Number of channels
Length=size(y,1);
Fmax = fs_downsample/2;
nfft=fs_downsample; %number of frequency bins

[w, Am, Cm, sbc, fpe, th] = arfit(y, 1, 40, 'sbc'); % ---> ARFIT toolbox
[tmp,p_opt] = min(sbc); % Optimum order for the MVAR model


% Create a cell array containing signal chunks
sigOut = {};


while( start_point + targetLength - 1 < Length )

       seg = y(start_point:start_point+targetLength-1,:);

       mask1(NumberChunks,:) = downmask(start_point:start_point+targetLength-1);
       seg = seg.*repmat(hamming(length(seg)),1,CH);
       [A_ST,RCF,C_ST] = mvar(seg, p_opt, Mode1); % ---> BioSig toolbox
       [PDC_ST(:,:,:,NumberChunks), DTF_ST(:,:,:,NumberChunks)] = PDC_DTF_matrix(A_ST,p_opt,fs_downsample,Fmax,nfft);
       start_point = start_point + (targetLength-overlap);
       NumberChunks = NumberChunks + 1;
end

    DTF_meanFrequency= squeeze(mean(DTF_ST, 3));

  %%reduce dimensionality from 2D matrix per frequency to 1D matrix per
  %%segment
    for n=1:size(DTF_meanFrequency,3) %identifies connectivity per segment
        CountFrequenyPoints=1;
        for j=1:size(DTF_meanFrequency,1) %identifies column of connectivity matrix
            for k=1:size(DTF_meanFrequency,2) %identifies row of connectivity matrix
                Reduced_DTF(CountFrequenyPoints,n)=DTF_meanFrequency(k,j,n);
                CountFrequenyPoints=CountFrequenyPoints+1;
            end
        end
    end

    for i=1:(NumberChunks-1)
        sigOut{i,1}= Reduced_DTF(:,i); %DTF per each chunk of signal (576x1) for each segment(CH*CH)

        v(i,:) = countcats(mask1(i,:));

        HighestCategory(i)=v(i,1);
        Category(i) = 1;

        for j=1:size(v,2)
            if (v(i,j)>HighestCategory(i))
                HighestCategory(i)=v(i,j);
                Category(i) = j;
            end
        end

%     lblOut{i,2} = Labels';

    end

% Create a cell array containing mask chunks
Labels= categorical (Category, [1 2 3 4 5 6 7], {'Fist' 'Flexion' 'Extension' 'PinchIndex' 'PinchMiddle' 'PinchRing' 'PinchSmall'});


lblOut = num2cell(Labels',2);
    
% Output a two-column cell array with all chunks
Tsds = [sigOut,lblOut];
end



    

