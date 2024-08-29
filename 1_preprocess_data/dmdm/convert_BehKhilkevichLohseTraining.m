function convert_BehKhilkevichLohseTraining(matfile, concat, loc_suffix)
% CONVERT_BEHKURODA  Convert Khilkevich and Lohse's training behavior data 
%                    in an Ashwood's GLM-HMM-friendly format
%
% This function uses npy-matlab to convert from .mat to .npy
% https://github.com/kwikteam/npy-matlab

    arguments
        matfile (1,1) string;
        concat (1,1) logical;
        loc_suffix (1,1) string = 'default';
    end

    dmdm_data_path = fullfile(fileparts(mfilename('fullpath')) + ...
                              "/../../data/dmdm/");
    if ~exist(dmdm_data_path, 'dir')
        mkdir(dmdm_data_path)
    end

    % Load .mat and list animals
    % This tends to take a while - as behav data is quite large
    fprintf('Loading %s ...\n', matfile);
    % disp(['Loading ' matfile, ' ...']);
    AllBehav = load(matfile); 

    if ~contains(fieldnames(AllBehav),'fsm')
        error(['Error. ', matfile, 'needs to be in final fsm format.']) 
    end

    animals = unique({AllBehav.fsm.participant});
    disp(['Found ', int2str(numel(animals)), ' animals.']);

    % Create directory for saving converted data:
    [~,orifname] = fileparts(matfile);
    % fname = erase(orifname,"_fsm");
    % saving_location = dmdm_data_path + fname + "/Subjects/";
    if strcmp(loc_suffix, 'default')
        saving_location = dmdm_data_path + 'dataAllMiceTraining' + "/Subjects/";
    else
        saving_location = dmdm_data_path + 'dataAllMiceTraining' + '_' + loc_suffix + "/Subjects/";
    end
    if ~exist(saving_location, 'dir')
        mkdir(saving_location)
    end

    Data = AllBehav.fsm;

    if ~concat
        %% Run a for loops for all animals
        for k=1:length(animals)
            if ~exist(saving_location + animals{k}, 'dir')
                mkdir(saving_location + animals{k})
            end
            subData = Data(strcmp({Data.participant}, animals{k}));
            % List sessions
            sessions = unique({subData.session});
    
            for s = 1:length(sessions)
                clear dmdm_trials
                if ~exist(saving_location + animals{k} + '/' + sessions{s}, 'dir')
                    mkdir(saving_location + animals{k} + '/' + sessions{s})
                end
                disp(sessions{s});
    
                fsm = subData(strcmp({subData.session}, sessions{s}));
    
                % trial outcome
                dmdm_trials.outcome = ConvertTrialOutcome({fsm.trialoutcome});
                writeNPY(dmdm_trials.outcome, saving_location + animals{k} + ...
                         '/' + sessions{s} + '/' + '_dmdm_trials.outcome.npy');   
    
                % change size
                % dmdm_trials.changesize = [fsm.Stim2TF];
                dmdm_trials.changesize = log2([fsm.Stim2TF]);
                writeNPY(dmdm_trials.changesize, saving_location + animals{k} + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.changesize.npy');
    
                % hazard block
                dmdm_trials.hazardblock = ConvertTrialHazardBlock_human({fsm.hazard});
                % still use ConvertTrialHazardBlock_human b/c they have probe trials here
                writeNPY(dmdm_trials.hazardblock, saving_location + animals{k} + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.hazardblock.npy');
    
                % reaction time
                dmdm_trials.reactiontimes = [fsm.RTbaseline];
                writeNPY(dmdm_trials.reactiontimes, saving_location + animals{k} + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.reactiontimes.npy');
    
                % change onset
                dmdm_trials.stimT = [fsm.stimT];
                writeNPY(dmdm_trials.stimT, saving_location + animals{k} + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.stimT.npy');

                % trial num
                dmdm_trials.trial = [fsm.trial];
                writeNPY(dmdm_trials.trial, saving_location + animals{k} + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.trial.npy');
            end
        end
    else
        if ~exist(saving_location + 'M_All', 'dir')
            mkdir(saving_location + 'M_All')
        end
        for k=1:length(animals)
            subData = Data(strcmp({Data.participant}, animals{k}));
            % List sessions
            sessions = unique({subData.session});

            for s = 1:length(sessions)
                clear dmdm_trials
                if ~exist(saving_location + 'M_All' + '/' + sessions{s}, 'dir')
                    mkdir(saving_location + 'M_All' + '/' + sessions{s})
                end
                disp(sessions{s});

                fsm = subData(strcmp({subData.session}, sessions{s}));
         
                % trial outcome
                dmdm_trials.outcome = ConvertTrialOutcome({fsm.trialoutcome});
                writeNPY(dmdm_trials.outcome, saving_location + 'M_All' + ...
                         '/' + sessions{s} + '/' + '_dmdm_trials.outcome.npy');   
    
                % change size
                dmdm_trials.changesize = log2([fsm.Stim2TF]);
                writeNPY(dmdm_trials.changesize, saving_location + 'M_All' + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.changesize.npy');
    
                % hazard block
                dmdm_trials.hazardblock = ConvertTrialHazardBlock_human({fsm.hazard});
                % still use ConvertTrialHazardBlock_human b/c they have probe trials here
                writeNPY(dmdm_trials.hazardblock, saving_location + 'M_All' + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.hazardblock.npy');
    
                % reaction time
                dmdm_trials.reactiontimes = [fsm.RTbaseline];
                writeNPY(dmdm_trials.reactiontimes, saving_location + 'M_All' + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.reactiontimes.npy');
    
                % change onset
                dmdm_trials.stimT = [fsm.stimT];
                writeNPY(dmdm_trials.stimT, saving_location + 'M_All' + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.stimT.npy');

                % trial num
                dmdm_trials.trial = [fsm.trial];
                writeNPY(dmdm_trials.trial, saving_location + 'M_All' + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.trial.npy');
            end
        end
    end
end