function new_arr = ConvertTrialOutcome(arr)
    new_arr = NaN(1, numel(arr));
    for i = 1:numel(arr)
        switch arr{i}
            case 'Hit'
                new_arr(i) = 1;
            case 'FA'
                new_arr(i) = 2;
            case 'Miss'
                new_arr(i) = 0;
            case 'abort'
                new_arr(i) = 3;
            otherwise
                new_arr(i) = NaN;
        end
    end
end