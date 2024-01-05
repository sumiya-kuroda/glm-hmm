function convert_BehLohse(matfile)
% CONVERT_BEHLOHSE  Convert Lohse's behavior data in an Ashwood's 
%                   GLM-HMM-friendly format
%
% This function uses npy-matlab to convert from .mat to .npy
% https://github.com/kwikteam/npy-matlab

    dmdm_data_path = fullfile(fileparts(mfilename('fullpath')) + ...
                              "/../../data/dmdm/");
    if ~exist(dmdm_data_path, 'dir')
        mkdir(dmdm_data_path)
    end

    % Load .mat and list animals
    % This tends to take a while - as behav data is quite large
    disp(['Loading ', matfile, ' ...']);
    AllBehav = load(matfile); 
    tmp = fieldnames(AllBehav);
    animals = CellStartWith(tmp, 'x');
    disp(['Found', numel(animals), ' animals in ', matfile]);

    % Create directory for saving converted data:
    [~,fname] = fileparts(matfile);
    saving_location = dmdm_data_path + fname + "/converted_to_npy/";
    if ~exist(saving_location, 'dir')
        mkdir(saving_location)
    end

    %% Run a for loops for all animals
    for k=1:length(animals)
        if ~exist(saving_location + animals{k}, 'dir')
            mkdir(saving_location + animals{k})
        end
        subData = AllBehav.(animals{k});
        % List sessions
        sessions = fieldnames(subData);

        for s = 1:length(sessions)
            clear dmdm_trials
            if ~exist(saving_location + animals{k} + '/' + sessions{s}, 'dir')
                mkdir(saving_location + animals{k} + '/' + sessions{s})
            end
            disp(sessions{s});

            fsm=subData.(sessions{s}).behav_data.trials_data_exp;
            % NI=subData.(sessions{s}).NI_events;
            % Video=subData.(sessions{s}).Video; 

            % trial outcome
            dmdm_trials.outcome = ConvertTrialOutcome({fsm.trialoutcome});
            writeNPY(dmdm_trials.outcome, saving_location + animals{k} + ...
                     '/' + sessions{s} + '/' + '_dmdm_trials.outcome.npy');
            
            % change size
            dmdm_trials.changesize = log2([fsm.Stim2TF]);
            writeNPY(dmdm_trials.changesize, saving_location + animals{k} + ...
                     '/' + sessions{s} + '/' + '_dmdm_trials.changesize.npy');

            % hazard block
            dmdm_trials.hazardblock = ConvertTrialHazardBlock({fsm.hazardblock});
            writeNPY(dmdm_trials.hazardblock, saving_location + animals{k} + ...
                     '/' + sessions{s} + '/' + '_dmdm_trials.hazardblock.npy');

            % reaction time
            dmdm_trials.reactiontimes = ConvertReactionTime(fsm);
            writeNPY(dmdm_trials.reactiontimes, saving_location + animals{k} + ...
                     '/' + sessions{s} + '/' + '_dmdm_trials.reactiontimes.npy');

                % if you want to use motion onset
                % Just be careful that some trials have no motiononset estimation
                % Baseline_ON_times = NI.Baseline_ON.rise_t;
                % Change_ON_times = NI.Change_ON.rise_t;
                % MotionOnsets=Video.MotionOnsetTimes;
                % RTs_FromBase=MotionOnsets-Baseline_ON_times;
                % RTs_FromChange=MotionOnsets-Change_ON_times;

            % change onset
            dmdm_trials.stimT = [fsm.stimT];
            writeNPY(dmdm_trials.stimT, saving_location + animals{k} + ...
                     '/' + sessions{s} + '/' + '_dmdm_trials.stimT.npy');

            % Future implementation: hazard probe, TF, orientation,
            % trialtype
        end

    end
end