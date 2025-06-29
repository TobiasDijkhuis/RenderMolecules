#!/usr/bin/env bash

usage="$(basename "$0") [-h] directory/to/env [path/to/blender/exe] -- program to add render_molecules and skimage to the Blender python path.

where:
    -h                    show this help text
    directory/to/env      directory of environment
    path/to/blender/exe   assd"

# Get directory of this file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# render_molecules directory
render_molecules_path="$SCRIPT_DIR/render_molecules"

is_root () {
    if [[ "$(whoami)" == "root" ]]; then
        true
    else
        false
    fi
}

echo $1
if [[ $1 == "-h" ]] || [[ $1 == "--help" ]]; then
    echo "$usage"
    exit
fi

if [[ $1 != "" ]]; then
    # Path to environment that contains scikit-image
    echo $1
fi

if [[ $2 -eq "" ]]; then
    # Assume that Blender is registered
    # Trying to find blender executable
    blenderExecutable=$(which blender)
    
    if [ $? -eq 1 ]; then
        echo "Could not find executable 'blender'. If it is not registered, please provide the path to the executable as the second argument"
        if is_root; then
            echo "Ran as root. If this script can find the blender executable as non-root,"
            echo "i.e. just running without the sudo in front, Blender probably is not registered for the sudo-user."
            echo "Try rerunning the script while using the normal user PATH by running:"
            echo "sudo -E env \"PATH=\$PATH\" ./install.sh"
            exit
        fi
        echo "Exiting"
        exit
    fi
    
    echo "Found blender executable at $blenderExecutable"
    
    if [[ -L "$blenderExecutable" ]]; then
        echo "$blenderExecutable is a symlink. Trying to resolve it"
        blenderExecutable=$(readlink -f $blenderExecutable)
        echo "It points to: $blenderExecutable"
    fi
else
    blenderExecutable=$2
    if [[ ! -f $blenderExecutable ]]; then
        echo "$blenderExecutable is not a file. Exiting"
    fi 
fi
blenderDirectory=$(dirname $blenderExecutable)

# Find the directory that contains the 'script' directory
scriptDir=$(find $blenderDirectory/* -depth -maxdepth 1 -name \*"scripts"\* -and -type d)
moduleDir="$scriptDir/modules"

echo "Making symlink to render_molecules directory"
if [[ -L "$moduleDir/render_molecules" ]]; then
    echo "Link to render_molecules directory already exists."
else
    echo "Trying to create symlink"
    if [ -w $moduleDir ]; then
        ln -s $render_molecules_path $moduleDir/
    else
        if ! is_root; then
            echo "Creating symlink in the module directory requires root access, but install script was not run with sudo. Please rerun"
            exit
        fi
        ln -s $render_molecules_path $moduleDir/
    fi
    echo "Successfully made symlink to render_molecules directory"
fi

echo "Testing imports. If this gives a ModuleNotFoundError, it has not worked."
pythonExpr=$"import render_molecules, skimage"
testCommand="$blenderExecutable -b --python-expr \"$pythonExpr\""
echo $testCommand
($testCommand)

if ($testCommand) 2>&1 | grep "Error"; then
    echo "Imports failed"
else
    echo "Imports succeeded"
fi
# blenderExecutable -b --python-expr "import render_molecules; import skimage" 2>&1 | grep "ModuleNotFoundError"
