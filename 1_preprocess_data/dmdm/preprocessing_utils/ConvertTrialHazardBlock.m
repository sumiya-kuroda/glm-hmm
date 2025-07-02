function new_arr = ConvertTrialHazardBlock(arr)
    new_arr = NaN(1, numel(arr));
    for i = 1:numel(arr)
        switch arr{i}
            case 'late'
                new_arr(i) = 1;
            case 'early'
                new_arr(i) = 0;
            otherwise
                new_arr(i) = NaN;
        end
    end
end