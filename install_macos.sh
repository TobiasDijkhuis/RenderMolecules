#!/usr/bin/bash

usage="$(basename "$0") [-h/--help] [path/to/blender/exe] -- program to add render_molecules and skimage to the Blender python path.

where:
    -h/--help             show this help text
    path/to/blender/exe   assd"

# Get directory of this file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# render_molecules directory
render_molecules_path="$SCRIPT_DIR/render_molecules"

# Function to figure out if the command was run with 'sudo'
is_root () {
    if [[ "$(whoami)" == "root" ]]; then
        true
    else
        false
    fi
}


# 
if [[ $1 == "-h" ]] || [[ $1 == "--help" ]]; then
    echo "$usage"
    exit
fi


if [[ $1 == "" ]]; then
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
else
    blenderExecutable=$2
    if [[ ! -f $blenderExecutable ]]; then
        echo "$blenderExecutable is not a valid filepath. Exiting"
        exit
    fi 
fi

# Resolve path to blender executable if a symlink was provided.
# 'which' gives the symlink, at least on my system.
if [[ -L "$blenderExecutable" ]]; then
    echo "Path to blender executable is a symlink. Trying to resolve it"
    blenderExecutable=$(readlink -f $blenderExecutable)
    echo "It points to: $blenderExecutable"
fi
blenderDirectory="../"$(dirname $blenderExecutable)
echo $blenderDirectory

# Find the directory that contains the 'script' directory. This depends on the blender version. In my case, it is 4.2
scriptDir=$(find $blenderDirectory/* -depth -maxdepth 2 -name \*"scripts"\* -and -type d)
echo $scriptDir


moduleDir="$scriptDir/modules"
if [[ ! -d "$moduleDir" ]]; then
    echo "$moduleDir directory does not exist yet. Trying to create it."
    mkdir "$moduleDir"
    echo "Successfully created directory $moduleDir"
fi

safe_symlink_at_dir () {
    echo "Making symlink to directory $1"
    directoryname=$(basename $1)
    if [[ -L "$2/$directoryname" ]]; then
        echo "Link $2/$directoryname already exists."
        return 0
    else
        echo "Trying to create symlink"
        if [[ -w "$2" ]]; then
            ln -s $1 $2/
        else
            if ! is_root; then
                echo "Creating symlink in $2 requires root access, but install script was not run with sudo. Please rerun"
                return 1
            fi
            ln -s $1 $2/
        fi
        echo "Successfully made symlink to $1 directory"
        return 0
    fi
}

# Make symlink to render_molecules directory
safe_symlink_at_dir $render_molecules_path $moduleDir

get_site_packages_dir () {
    echo $(python -c 'import site; print(site.getsitepackages()[0])')
}
site_packages_dir=$(get_site_packages_dir)

# Make symlink to skimage directory in the environment
safe_symlink_at_dir $site_packages_dir/skimage $moduleDir

# Test imports to see if they worked
echo "Testing imports."
imports=$"import render_molecules;import skimage"
stdout=$($blenderExecutable -b --python-exit-code 144 --python-expr "$imports" 2>&1)
if [[ $? == 144 ]] || [[ ${#stderr} != 0 ]]; then
    echo "Imports failed. Command output:"
    echo "$stdout"
else
    echo "Imports succeeded"
fi
