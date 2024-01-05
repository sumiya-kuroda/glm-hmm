#!/bin/bash
# https://gist.github.com/rragundez/40025e94b91c147a709b8dda300d5b5f
declare opt
declare OPTARG
declare OPTIND
OPTIND=1

usage="$(basename "$0") [-h] [-i input] [-f format]
Convert behavior data
where:
    -h  show this help text
    -i  input
    -f  which dmdm project format
"

BASE="$( cd "$( dirname "$BASH_SOURCE" )" && pwd -P )"

while getopts ':hi:f:' opt; do
  case "${opt}" in
    h) echo "$usage" >&2; return;;
    i) INPUT=$OPTARG;;
    f) FORMAT=$OPTARG;;
  esac
done

# mandatory arguments 
# [ ! "$INPUT" ] || [ ! "$FORMAT" ]
if [ ! "$INPUT" ]; then
  echo "Error: arguments -i and -f must be provided"; return;
fi

matlab -nodisplay -nosplash -nodesktop -r "cd('$BASE'); \
                                           addpath(genpath('.')); \
                                           convert_BehLohse('$INPUT'); \
                                           exit"