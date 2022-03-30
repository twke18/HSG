% Please follow BSR section in the README.md before running the script.
clear all; close all; clc;

th = 0.25;
nSegTh = 48; % number of maximum segments
isUcm = 1;
isSeg = 1;

edge_dir = './voc12/edge/';

ucm_dir = ['./voc12/ucm/'];
if ~exist(ucm_dir, 'dir')
    mkdir(ucm_dir);
end

seg_dir = ['./voc12/rf_' num2str(th) '_' num2str(nSegTh) '/'];
if ~exist(seg_dir, 'dir')
    mkdir(seg_dir);
end

edge_list = dir([edge_dir '*.mat']);


% Add toolbox path.
addpath(genpath('./BSR/toolbox/'));

% Add MCG path.
addpath(genpath('./BSR/mcg-2.0/full/src/ucms/'));
addpath(genpath('./BSR/mcg-2.0/full/lib/'));

for i = 1:numel(edge_list)
    filename = edge_list(i).name;

    % Load predicted contours from HED.
    predict = load([edge_dir filename]);
    E = single(predict.predict);

    if isUcm
        % Calculate orientations of contours.
        [Ox,Oy] = gradient2(convTri(E,4));
        [Oxx,~] = gradient2(Ox);
        [Oxy,Oyy] = gradient2(Oy);
        O = mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);

        % Perform oriented watershed transform.
        [owt2, superpixels] = contours2OWT(E, O);

        % Perform spectral globalization.
        sPb = spectralPb_fast(owt2);
        sPb = (sPb - min(sPb(:))) / (max(sPb(:)) - min(sPb(:)));

        % Calculate ucm2.
        ucm2 = double(ucm_mean_pb(owt2 + sPb, superpixels));
        save([ucm_dir filename(1:end-3) 'mat'], 'ucm2');
    end

    if isSeg
        % Calculate the final segmentation given a threshold.
        ucm_data = load([ucm_dir filename(1:end-3) 'mat'], 'ucm2');
        ucm2 = ucm_data.ucm2;
        for d_th = 0:0.01:1.0
            labels2 = bwlabel(ucm2 <= (th + d_th));
            labels = labels2(2:2:end, 2:2:end);
            if numel(unique(labels)) <= nSegTh
                break;
            end
        end
        assert(numel(unique(labels(:))) < 256);

        labels = uint16(labels);
        fprintf('image: %d, num. of segs: %d\n', i, numel(unique(labels)));
        imwrite(labels, [seg_dir filename(1:end-3) 'png']);
    end
end
