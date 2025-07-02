function new_arr = CellStartWith(arr, letter)
    new_arr = arr(strncmp(arr, letter, numel(letter)));
end