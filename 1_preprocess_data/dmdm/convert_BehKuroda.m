function convert_BehKuroda(matfile, concat, loc_suffix)
% CONVERT_BEHKURODA  Convert Kuroda's behavior data in an Ashwood's 
%                    GLM-HMM-friendly format
%
% This function uses npy-matlab to convert from .mat to .npy
% https://github.com/kwikteam/npy-matlab

    arguments
        matfile; % (1,1) string mess up disp
        concat (1,1) logical;
        loc_suffix string = 'default';
    end

    dmdm_data_path = fullfile(fileparts(mfilename('fullpath')) + ...
                              "/../../data/dmdm/");
    if ~exist(dmdm_data_path, 'dir')
        mkdir(dmdm_data_path)
    end

    % Load .mat and list animals
    % This tends to take a while - as behav data is quite large
    % disp('Loading ' + matfile + ' ...');
    AllBehav = load(matfile); 
    animals = AllBehav.animals;
    disp(['Found ' int2str(numel(animals)) ' animals in ' matfile]);

    % Create directory for saving converted data:
    [~,fname] = fileparts(matfile);
    % saving_location = dmdm_data_path + fname + "/Subjects/";
    if strcmp(loc_suffix, 'default')
        saving_location = dmdm_data_path + 'dataAllHumans' + "/Subjects/";
    else
        saving_location = dmdm_data_path + 'dataAllHumans' + loc_suffix + "/Subjects/";
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
                [dmdm_trials.outcome_withabort, ...
                    dmdm_trials.outcome_noabort] = ConvertTrialOutcome_human(fsm);
                writeNPY(dmdm_trials.outcome_withabort, saving_location + animals{k} + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.outcome.npy');
                writeNPY(dmdm_trials.outcome_noabort, saving_location + animals{k} + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.outcome_noabort.npy');            
                % change size
                dmdm_trials.changesize = log2([fsm.Stim2TF]);
                writeNPY(dmdm_trials.changesize, saving_location + animals{k} + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.changesize.npy');

                % hazard block
                dmdm_trials.hazardblock = ConvertTrialHazardBlock_human({fsm.hazard});
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

                % Future implementation: hazard probe, TF, orientation,
                % trialtype
            end

        end
    else
        if ~exist(saving_location + 'HAll', 'dir')
            mkdir(saving_location + 'H_All')
        end
        for k=1:length(animals)
            subData = Data(strcmp({Data.participant}, animals{k}));
            % List sessions
            sessions = unique({subData.session});

            for s = 1:length(sessions)
                clear dmdm_trials
                if ~exist(saving_location + 'HAll' + '/' + sessions{s}, 'dir')
                    mkdir(saving_location + 'HAll' + '/' + sessions{s})
                end
                disp(sessions{s});

                fsm = subData(strcmp({subData.session}, sessions{s}));

                % trial outcome
                [dmdm_trials.outcome_withabort, ...
                    dmdm_trials.outcome_noabort] = ConvertTrialOutcome_human(fsm);
                writeNPY(dmdm_trials.outcome_withabort, saving_location + 'HAll' + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.outcome.npy');
                writeNPY(dmdm_trials.outcome_noabort, saving_location + 'HAll' + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.outcome_noabort.npy');            
                
                % change size
                dmdm_trials.changesize = log2([fsm.Stim2TF]);
                writeNPY(dmdm_trials.changesize, saving_location + 'HAll' + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.changesize.npy');

                % hazard block
                dmdm_trials.hazardblock = ConvertTrialHazardBlock_human({fsm.hazard});
                writeNPY(dmdm_trials.hazardblock, saving_location + 'HAll' + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.hazardblock.npy');

                % reaction time
                dmdm_trials.reactiontimes = [fsm.RTbaseline];
                writeNPY(dmdm_trials.reactiontimes, saving_location + 'HAll' + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.reactiontimes.npy');

                % change onset
                dmdm_trials.stimT = [fsm.stimT];
                writeNPY(dmdm_trials.stimT, saving_location + 'HAll' + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.stimT.npy');

                % trial num
                dmdm_trials.trial = [fsm.trial];
                writeNPY(dmdm_trials.trial, saving_location + 'HAll' + ...
                        '/' + sessions{s} + '/' + '_dmdm_trials.trial.npy');
            end
        end
    end
end