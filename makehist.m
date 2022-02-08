% Input: name.wav.
% Output: histogram of the voice recording.
%[voice, FS] = audioread(name.wav’); % read the original/fake voice recording.
%nbins = 65536; % number of bins of the histogram.
%h = histogram(voice, nbins); % plot the histogram.
set(0,'DefaultFigureVisible','off')

myDir = 'D:\Data\Testing_data'; %gets directory
myFiles = dir(fullfile(myDir,'*.wav')); %gets all wav files in struct
train_dir = 'D:\\Data\\test_data\\hist_t_%d.jpg';
parfor k = 1:length(myFiles)
  baseFileName = myFiles(k).name;
  fullFileName = fullfile(myDir, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  [voice, FS] = audioread(fullFileName); % read the original/fake voice recording.
  nbins = 65536; % number of bins of the histogram.
  h = histogram(voice, nbins); % plot the histogram.
  
  savePath = sprintf(train_dir, k-1);
  fprintf(1, 'Now saving %s\n', savePath);
  
  saveas(h, savePath);
end