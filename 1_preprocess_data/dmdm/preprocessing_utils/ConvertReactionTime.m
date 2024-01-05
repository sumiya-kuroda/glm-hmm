function new_RT_arr = ConvertReactionTime(struct)
    % All RTs are from baseline onset except miss trials have NaN value
    new_RT_arr = NaN(1, numel(struct));
    stimT_arr = [struct.stimT];
    for i = 1:numel(struct)
        if ~isnan(struct(i).reactiontimes.FA)
            new_RT_arr(i)=struct(i).reactiontimes.FA;
        elseif ~isnan(struct(i).reactiontimes.RT)
            new_RT_arr(i)=struct(i).reactiontimes.RT + stimT_arr(i);
        elseif ~isnan(struct(i).reactiontimes.Ref)
            new_RT_arr(i)=struct(i).reactiontimes.Ref;
        elseif ~isnan(struct(i).reactiontimes.Miss)
            new_RT_arr(i)=NaN;
        elseif ~isnan(struct(i).reactiontimes.gray)
            new_RT_arr(i)=struct(i).reactiontimes.gray;
        elseif ~isnan(struct(i).reactiontimes.abort)
            new_RT_arr(i)=struct(i).reactiontimes.abort;
        end
    end
end
