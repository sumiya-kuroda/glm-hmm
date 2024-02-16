function [new_arr_withabort, new_arr_noabort] = ConvertTrialOutcome_human(struct)
    arr = {struct.trialoutcome};
    new_arr_noabort = NaN(1, numel(arr));
    for i = 1:numel(arr)
        switch arr{i}
            case 'Hit'
                new_arr_noabort(i) = 1;
            case 'FA'
                new_arr_noabort(i) = 2;
            case 'Miss'
                new_arr_noabort(i) = 0;
            case 'Abort'
                new_arr_noabort(i) = 3;
            case 'Ref'
                new_arr_noabort(i) = 4;
            otherwise
                new_arr_noabort(i) = NaN;
        end
    end

    new_arr_withabort = copy(new_arr_noabort);
    new_arr_withabort(struct.IsAborted) = 3; % replace with abort

end