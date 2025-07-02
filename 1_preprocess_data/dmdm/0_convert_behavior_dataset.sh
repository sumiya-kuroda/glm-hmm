#!/bin/bash
# https://gist.github.com/rragundez/40025e94b91c147a709b8dda300d5b5f
declare opt
declare OPTARG
declare OPTIND
OPTIND=1

usage="$(basename "$0") [-h] [-i input] [-f format] [-s suffix]
Convert behavior data from .mat to .npy
where:
    -h  show this help text
    -i  input
    -f  which dmdm project format: Choose from Lohse or Kuroda
    -s  suffix of saving location if any
"

BASE="$( cd "$( dirname "$BASH_SOURCE" )" && pwd -P )"

while getopts ':hi:f:s:' opt; do
  case "${opt}" in
    h) echo "$usage" >&2; return;;
    i) INPUT=$OPTARG;;
    f) FORMAT=$OPTARG;;
    s) SUFFIX=$OPTARG;;
  esac
done

# mandatory arguments 
if [ ! "$INPUT" ] || [ ! "$FORMAT" ]|| [ ! "$SUFFIX" ]; then
  echo "Error: arguments -i, -f, and -s must be provided"; return;
fi

if [ "$FORMAT" = "Lohse" ]; then
  matlab -nodisplay -nosplash -nodesktop -r "cd('$BASE'); \
                                            addpath(genpath('.')); \
                                            convert_BehLohse('$INPUT'); \
                                            exit"
elif [ "$FORMAT" = "KhilkevichLohseTraining" ]; then
  matlab -nodisplay -nosplash -nodesktop -r "cd('$BASE'); \
                                            addpath(genpath('.')); \
                                            convert_BehKhilkevichLohseTraining('$INPUT', false, 200, '$SUFFIX'); \
                                            exit"
elif [ "$FORMAT" = "Kuroda" ]; then
  matlab -nodisplay -nosplash -nodesktop -r "cd('$BASE'); \
                                            addpath(genpath('.')); \
                                            convert_BehKuroda('$INPUT', true); \
                                            exit"
else
  echo "Error: Format $FORMAT not supported"; return;
fi