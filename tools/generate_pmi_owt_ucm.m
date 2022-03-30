%% Detect boundaries
% you can control the speed/accuracy tradeoff by setting 'type' to one of the values below
% for more control, feel free to play with the parameters in setEnvironment.m
compile;
clc; clear;

%type = 'speedy'; % use this for fastest results
type = 'accurate_low_res'; % use this for slightly slower but more accurate results
%type = 'accurate_high_res'; % use this for slow, but high resolution results

thresh = 0.05; % larger values give fewer segments
nSegTh = 1024; % number of maximum segments
rootDir = '/home/twke/data/Cityscapes/';
matDir = '../ucm2_uint8/cityscapes/mats';
segDir = sprintf('../ucm2_uint8/cityscapes/pmi_%.2f_%d', thresh, nSegTh);
imgNames = textread('../text_files/cityscapes_train.txt', '%s');
imgNames= imgNames(1:3:end);

% pipeline
isEdge = 1;
isSeg = 1;
isRmvBnd = 1;

if exist(matDir) ~= 7
    mkdir(matDir);
end

if exist(segDir) ~= 7
    mkdir(segDir);
    mkdir([segDir '/gray']);
    mkdir([segDir '/color']);
end

for i = 1:numel(imgNames)
    imgName = fullfile(rootDir, imgNames{i});
    I = imread(imgName);
    if isEdge
        [E,E_oriented] = findBoundaries(I,type);
        E_ucm = contours2ucm_crisp_boundaries(mat2gray(E_oriented));
        [~, matName, ~] = fileparts(imgNames{i});
        matName = fullfile(matDir, matName);
        save(matName, 'E_ucm');
    end
    if isSeg
        if ~isEdge
            [~, matName, ~] = fileparts(imgNames{i});
            matName = fullfile(matDir, matName);
            matData = load(matName, 'data');
            E_ucm = matData.data;
        end
        [~, segName, ~] = fileparts(imgNames{i});

        for d_th = 0:0.01:1.0
            L = bwlabel(E_ucm <= (thresh + d_th));
            if numel(unique(L) <= nSegTh)
                break;
            end
        end
        L = uint16(L);

        % dilate to remove boundary.
        if isRmvBnd
            se = strel('square', 3);
            dL = imdilate(L, se);
            mask = dL > 0;
            assert(all(mask(:)));
            bndInds = find(L == 0);
            L(bndInds) = dL(bndInds);
        end

        L = imresize(L, 2, 'nearest');

        display(segName).
        display(numel(unique(L)));
        segLName = strcat(segName, '.png');
        segLName = fullfile(segDir, segLName);
        imwrite(L, segLName);
    end
end

