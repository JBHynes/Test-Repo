%% SSIMNETS toolbox demo
%
% The following code loads data from a planar center out task,
% computes 3-D Neuron similarity spaces and performs
% classification using a simple nearest neighbor classifier
% The 3D results are then graphically displayed. 
% 
%  You can change the parameters to see the effects of different:
%  1)  spiketrain time windows
%  2)  q values  (i.e. temporal sensitivity of analysis)
%  3)  perplexity (# of neigbouring neurons that influence the location of given neurons location in the relational map  )
%  Code requires SSIMS toolbox + helper functions:
%
% 
% Validated using MatLab R2016a
% SIMNETS @authors: Carlos Vargas-Irwin, Jacqueline Hynes,Brandman, D. M.,Zimmermann  
% for SSIMS details, see:
% Vargas-Irwin, C. E., Brandman, D. M., Zimmermann, J. B., Donoghue, J. P., & Black, M. J
% "Spike Train SIMilarity Space (SSIMS): A Framework for Single Neuron and Ensemble Data Analysis." (2014).
%
% @author Jacqueline Hynes
% Copyright (c) Jacqueline Hynes, Brown University. All rights reserved.
% Questions? Contact Jacqueline Hynes@Brown.edu

close all;
clear all;

%% Load data:
%  This dataset includes neural activity recorded from macaque motor cortex,
%  Event times, and information about movement direction for each trial.
load('SSIMNETS_demo_centeroutData');
%   spike_timestamps: cell array with spike train data from 103 units. Spike trains
%       are double arrays whose entries correspond to times of spike
%       occurences, in seconds, referenced to a common zero
%   instruction_cue: event times for 114 instruction cue events during the
%       recording. Double array, in seconds, referenced to the same common
%       zero as spikes.
%   go_cue: event times for 114 go cue events during the
%       recording. Double array, in seconds, referenced to the same common
%       zero as spikes.
%   start_of_movement: event times for 114 start of movement events during the
%       recording. Double array, in seconds, referenced to the same common
%       zero as spikes.
%   movement_direction: double vector of 114 entries, signifying the direction of
%       the movement corresponding to events in start_of_movement (in degrees)
%   pd: double vector of 103 entries, signifying the neurons preferred direction
%   of movement (0, 359). PDs were calculated by fitting a Von misses function to fr rates. 
%% Set parameters

% Select event types: instruction_cue, go_cue, or start_of_movement  
 events =   start_of_movement;  % NOTE: supplied preferred directions are specific to start_of_movement epoch
 
% Set the start of the time-window, relative to events, in s:
  start_offset = -.1;  

% total length of spiketrain windows, in s:
  win_len = 1;            

% number of dimensions for visualization of Neuron Relational Map:
  displaydim = 3;

% temporal accuracy parameter (from Victor & Purpura), when timeing in Sec,
% q = 0, rate code;  q = 10, 100 msec  ; q = 100, 10 ms,... i.e. 1/q = temporal sensitivity. 
  q =30;

% Perplexity value (specific to tSNE algorithm): # of neurons that influence a given neuron during dimensionality reduction 
% Should be < (# of neurons). Reccomended range: perp = [20:70];
   perp= 30 ;

% Max of search range for optimal k-means cluster number:
 crange =10 ;  

% Statistical Test: ~20 shuffles is minimum iteration # that can resonably be used. >50 iterations is statistically sufficient, >100 is best practice (but slow). 
shuffle = 1; % 1 for yes, 0 for no. 
iterations = 30;  
CIval=0.001;  
%% SSIMNETS: 
% 'Neuron X Neuron Correlations Matrix' will be passed to dimensionality reduction algoithm,
%  tsne. Each (ii,jj) element of NxN Corr matrix is the spearman corr of a pair of
% 'Event x Event spiketrain similarity matrices'. 

% This step can take some time (~ 10 min) using the plain MATLAB functions
% included, but it can be *tremendously* accelerated (~1 min) by compiling the
% toolbox functions as lined out in doc/INSTALL.md
pl=1;
it=1;

spike_timestamps = repmat(spike_timestamps(1:100), [1 100]);
pd = repmat(pd(1:100), [100 1]);

colsPD =  hsv(unique(pd+1)); 

cpu=zeros(it,20);

for nn = [500]
     
 for tr = 100
     
     for it=1:it
        % 1) Outputs the neuron correlation maxtrix ( ncmat) 
        tic
            [ ncmat, dmatnb ] = SSIMSNeuronRelCorr(spike_timestamps(1:nn), events(1:tr)+start_offset, win_len, q);

        cpuLOG(it,pl) = toc;
                                % IF 'tsne_PCA' ERROR:
                                % ncmat likely contains nans. NANs are the results of single-neuron distance
                                % matrices with all zero elements. To fix, choose option A,B,C at your own discretion:
                                % OPTION A: use 'distance correlation' instead of 'corr'.  
                                % OPTION B: Set nans to zero.
                                % OPTION C: Remove nans from Neuron x Neuron Array (excludes neurons that don't fire)
                                   removeNAN=1;
                                if removeNAN==1
                                    ncmat(isnan(ncmat))=0; % zero
                                elseif removeNAN==2
                                    rowNAN=find(all(isnan(ncmat),1));  % Remove all nans
                                    ncmat=removerows(ncmat','ind',rowNAN);
                                    ncmat=removerows(ncmat','ind',rowNAN); 
                                    pd(rowNAN)=[]; % Remove pds
                                end
   
   
        %2) Use t-SNE to project NXN matrix into a low dimensional space
%          fixed =  1-(ncmat/norm(ncmat));
            NSPACE = tsne_PCA(ncmat,[],displaydim,[],300);

            % Apply a varimax rotation for optimal plotting angle (viewing anlge can be
            % rotated after plotting, however). 
             NS_PC  = pca(NSPACE);
             NSPACE = NSPACE*NS_PC;


        %3) Cluster Detection using K-means:

            [s cindex centroids nclus ] = autokmeanscluster(5,NSPACE);
           subplot(2,2,1) 
           plotNT(NSPACE,pd(1:nn),'color', colsPD); 

        




     end
    pl=pl+1;  
   
      
    end     
         
end 
save cpuN
% 4) Shuffle test: 
if shuffle==1
    [ ncmatShift, shuffleSilh] = SSIMSNeuronRelCorr_shuffleTest(dmatnb, crange,perp, displaydim,CIval, 1000);
    SNETS.Shuffle = shuffleSilh;
end     
    
    % Add results to Data Structure 

    SNETS.NSPACE = NSPACE;
    SNETS.clusterindex = cindex;
    SNETS.centroids = centroids;
    SNETS.silhouette = s;
    SNETS.numclus = nclus;
    SNETS.NNcorr = ncmat;
    SNETS.distmat = dmatnb;
  
%% Vizualization / Plotting Data 

% Select represented preferred directions from full 360 degree color gradient 
    cols360 = hsv(360);  
    colsPD = cols360(sort(unique(pd))+1,:);
  
% 1a: Plot neurons labeled with their preferred directions 
    subplot(2,2,1)
    plotNT(SNETS.NSPACE,pd,'color', colsPD); 
   tt= toc
    title('Neuron space - preferred directions')
    colorbar; colormap hsv; caxis([0 359 ]);
     hcb = colorbar;
    title(hcb,'pd')
% 1b: Plot neurons with labeled with the k-means clusters indices 
    subplot(2,2,2) 
    plotNT(SNETS.NSPACE,SNETS.clusterindex)
    title('Neuron space - k-means clusters')
    legend({'cluster 1','cluster 2','cluster 3'})



% 2a: Silhouette optimal k-means cluster number search
    
    subplot(2,2,3)
    imagesc(SNETS.NNcorr);  
    title('Neuron Correlation Matrix')
    xlabel('neurons')
    ylabel('neurons')
   
    subplot(2,2,4)
    plot(2:max(crange), SNETS.silhouette(2:max(crange)),'ko-', 'linewidth',1.5);box off
    hold on; 
    xlabel('cluster #')
    ylabel('mean sil. values')   
    xlim([ 2 crange])
    set(gca,'fontsize',14)
     title('Silhouette Plot - showing peak (optimal) cluster number')
%     if shuffle==1
%          jbfill(2:crange,SNETS.Shuffle.CI(1,2:crange) , SNETS.Shuffle.CI(2,2:crange), 2:crange);
%          plot(2:max(crange), SNETS.Shuffle.mu(2:max(crange)),'r-', 'linewidth',1.5);box off
%    
%     end
%     


