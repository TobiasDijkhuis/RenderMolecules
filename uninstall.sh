#!/usr/bin/bash

usage="$(basename "$0") [-h/--help] [path/to/blender/exe] -- program to remove render_molecules and skimage from the Blender python path.

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
blenderDirectory=$(dirname $blenderExecutable)

# Find the directory that contains the 'script' directory. This depends on the blender version. In my case, it is 4.2
scriptDir=$(find $blenderDirectory/* -depth -maxdepth 1 -name \*"scripts"\* -and -type d)

moduleDir="$scriptDir/modules"

safe_remove_symlink_at_dir () {
    echo "Removing symlink to directory $1"
    directoryname=$(basename $1)
    if [[ ! -L "$2/$directoryname" ]]; then
        echo "No symlink $2/$directoryname"
        return 1
    else
        echo "Trying to remove symlink"
        if [[ -w "$2" ]]; then
            rm $2/$directoryname
        else
            if ! is_root; then
                echo "Removing symlink in $2 requires root access, but uninstall script was not run with sudo. Please rerun"
                return 1
            fi
            rm $2/$directoryname
        fi
        echo "Successfully removed symlink to $1 directory"
        return 0
    fi
}

# Make symlink to render_molecules directory
safe_remove_symlink_at_dir $render_molecules_path $moduleDir

get_site_packages_dir () {
    echo $(python -c 'import site; print(site.getsitepackages()[0])')
}
site_packages_dir=$(get_site_packages_dir)

# Make symlink to skimage directory in the environment
safe_remove_symlink_at_dir $site_packages_dir/skimage $moduleDir

