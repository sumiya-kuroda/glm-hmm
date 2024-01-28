%% Run GLM with mnrfit
clear
close all;
addpath 'C:\Users\Sumiya\Workstation2023\GeneralFunctions\npy-matlab\npy-matlab'
N_initializations = 10;
model = 'nominal';
fold = 0;
rng(65);
load fisheriris
X = meas(51:end,:);
y = strcmp('versicolor',species(51:end));
% binaryindex = find(strcmp(species, 'setosa'));
% species(binaryindex,:) = [];
% meas(binaryindex,:) = [];
b = glmfit(X,y,'binomial','link','logit')
b1 = mymnrfit(X,y,'model',model);


filenameX ='C:\Users\Sumiya\Workstation2023\glm-hmm\data\dmdm\dataAllMice\data_for_cluster_rt\test_normalized_y_inpt.npy';
filenameY ='C:\Users\Sumiya\Workstation2023\glm-hmm\data\dmdm\dataAllMice\data_for_cluster_rt\test_master_y.npy';
filenameSession ='C:\Users\Sumiya\Workstation2023\glm-hmm\data\dmdm\dataAllMice\data_for_cluster_rt\test_master_session.csv';
filenameSessionLookUPTable ='C:\Users\Sumiya\Workstation2023\glm-hmm\data\dmdm\dataAllMice\data_for_cluster_rt\all_animals_concat_session_fold_lookup.csv';

X = readNPY(filenameX);
Y = readNPY(filenameY);
Session = readtable(filenameSession,"Delimiter",",",ReadVariableNames=false);
SessionLookUPTable = readtable(filenameSessionLookUPTable,"Delimiter",",",ReadVariableNames=false);

sessions_to_keep = SessionLookUPTable(SessionLookUPTable.Var2 ~= fold,1).Var1;
idx_this_fold = find(ismember(Session.Var1, sessions_to_keep));

this_inpt_y = X(idx_this_fold,:);
this_y =  Y(idx_this_fold);

mymnrfit(this_inpt_y,this_y+1,'model',model)

for i=1:N_initializations
    % w = rand(6, 1);
    % B{i,1} = mymnrfit(this_inpt_y,this_y + 1,'model',model, IterationLimit=1000, Tolerance=1e-1);
    [B{i,1},dev{i,1},stats{i,1}] = mymnrfit(meas,species,'model',model, IterationLimit=10000, Tolerance=1e-6);
    results{i,1} = sum(stats{1,1}.residd);
    % [B,dev,stats]=
    % MnrMdl{i}=fitmnr(this_inpt_y,this_y,'ModelType',model);
    figure; plot(B{i});
end

% load fisheriris
% MnrModel = fitmnr(meas,species)