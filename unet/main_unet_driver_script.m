%  main_unet_driver_script
%
% MATLAB U-Net training script example
%
%   This is example calls a custom function provided by the Mathworks,
%   "createUnet()", which allows one to construct a unet of arbitrary size.
%
%   The code below demostrates a simple training scenerio and calls the 
%   "createUnet()" function.  
%
%  Part of this script follows the flow at this link; however, the custom
%   "createUnet()" function allows you to do more customization than what
%   is shown at the link
%       https://www.mathworks.com/help/vision/ref/unetlayers.html
%
% W. C. Walton  3/19/2021
%---------------------------------------------------------

% Load an image datastore of digit images

dataSetDir = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
imageDir = fullfile(dataSetDir,'trainingImages');
labelDir = fullfile(dataSetDir,'trainingLabels');

% Create imageDatasore object
imds = imageDatastore(imageDir);

% Define class names and IDs
classNames = ["triangle","background"];
labelIDs   = [255 0];

% Create pixelLabelDatastore to store ground truth pixel labels
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

for ii = 1:2  % <-- Just to see a couple of the input training pairs
    
    x1 = imread(imds.Files{ii});
    x2 = imread(pxds.Files{ii});
    
    figure(1);
    subplot(1,2,1);imshow(x1);title('Input');
    subplot(1,2,2);imshow(x2);title('Target');
    pause(0.5);
end

% Get the image dimensions
x1 = imread(imds.Files{ii});
[nr,nc] = size(x1);

% Create the datastore with the input and target images
ds = combine(imds,pxds);


% Specify the network architecture

lgraph = createUnet(nr,nc);
figure; plot(lgraph);pause(0.2);


%options = trainingOptions('adam');
options = trainingOptions('sgdm',...
    'MiniBatchSize',1,...
    'MaxEpochs',5,...   % Could try fewer or more Epochs
    'InitialLearnRate',1e-3,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',20,...
    'Shuffle','every-epoch'); 

net = trainNetwork(ds,lgraph,options);

for ii = 1:5  % <-- Just to see a few example outputs
    
    x1 = imread(imds.Files{ii});
    
    % Here, just to do a quick test of the network, use the same input used
    % for training (OR, add some corruption in order to simulate a different
    % image for testing.
    x1test = x1;
    %x1test = imgaussfilt(x1,0.1);  % Corrupted input
   
    % Target image
    x2 = imread(pxds.Files{ii});
    
    % Run the network
    x_output = predict(net,x1test);
    
    % Show the input and output images
    figure(1);
    subplot(2,2,1);imshow(x1test);title('Test input');
    subplot(2,2,2);imshow(x2,[]);title('Target');
    subplot(2,2,3);imshow(x_output(:,:,1),[]);title('Network otuput 1');
    subplot(2,2,4);imshow(x_output(:,:,2),[]);title('Network otuput 2');
   
    % Hit a key
    keyboard;
end

% Example training round

% |========================================================================================|
% |  Epoch  |  Iteration  |  Time Elapsed  |  Mini-batch  |  Mini-batch  |  Base Learning  |
% |         |             |   (hh:mm:ss)   |   Accuracy   |     Loss     |      Rate       |
% |========================================================================================|
% |       1 |           1 |       00:00:00 |       14.06% |      12.1027 |          0.0010 |
% |       1 |          50 |       00:00:01 |       94.43% |       0.1902 |          0.0010 |
% |       1 |         100 |       00:00:03 |       96.39% |       0.0991 |          0.0010 |
% |       1 |         150 |       00:00:04 |       95.12% |       0.1169 |          0.0010 |
% |       1 |         200 |       00:00:06 |       99.51% |       0.0219 |          0.0010 |
% |       2 |         250 |       00:00:07 |       97.85% |       0.0497 |          0.0010 |
% |       2 |         300 |       00:00:09 |       98.54% |       0.0531 |          0.0010 |
% |       2 |         350 |       00:00:10 |       92.29% |       0.1588 |          0.0010 |
% |       2 |         400 |       00:00:12 |       96.68% |       0.0711 |          0.0010 |
% |       3 |         450 |       00:00:13 |       97.85% |       0.0481 |          0.0010 |
% |       3 |         500 |       00:00:15 |       98.83% |       0.0238 |          0.0010 |
% |       3 |         550 |       00:00:16 |       99.02% |       0.0320 |          0.0010 |
% |       3 |         600 |       00:00:18 |       99.22% |       0.0289 |          0.0010 |
% |       4 |         650 |       00:00:19 |       99.51% |       0.0169 |          0.0010 |
% |       4 |         700 |       00:00:21 |       93.16% |       0.1448 |          0.0010 |
% |       4 |         750 |       00:00:22 |       98.14% |       0.0431 |          0.0010 |
% |       4 |         800 |       00:00:24 |       96.58% |       0.0648 |          0.0010 |
% |       5 |         850 |       00:00:25 |       99.61% |       0.0088 |          0.0010 |
% |       5 |         900 |       00:00:27 |       96.58% |       0.0631 |          0.0010 |
% |       5 |         950 |       00:00:28 |       99.02% |       0.0355 |          0.0010 |
% |       5 |        1000 |       00:00:30 |       99.61% |       0.0074 |          0.0010 |
% |========================================================================================|