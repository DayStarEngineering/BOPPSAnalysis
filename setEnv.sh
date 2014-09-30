#!/bin/bash

# Update path:
if [ -n "$PATH" ]; then
        PATH=`pwd`/mit_linux:`pwd`/fsmcontroller:`pwd`/clm:${PATH}
else
        PATH=`pwd`/mit_linux:`pwd`/fsmcontroller:`pwd`/clm
fi

# Update library path (different for mac and linux...):
OS=`uname`
# It would be nice to do this more elegantly...
if [[ "$OS" == 'Darwin' ]]; then
    echo "You're on a mac! So your dynamic library variable is: DYLD_LIBRARY_PATH"
    if [ -n "$DYLD_LIBRARY_PATH" ]; then
        $DYLD_LIBRARY_PATH=`pwd`/predictiveFilter:`pwd`/Centroid:${$DYLD_LIBRARY_PATH}
    else
        export DYLD_LIBRARY_PATH=`pwd`/predictiveFilter:`pwd`/Centroid
    fi
else
    if [ -n "$LD_LIBRARY_PATH" ]; then
        $LD_LIBRARY_PATH=`pwd`/predictiveFilter:`pwd`/Centroid:${$LD_LIBRARY_PATH}
    else
        export LD_LIBRARY_PATH=`pwd`/predictiveFilter:`pwd`/Centroid
    fi
fi

# Ipython Notebook Set-up:
export IPYTHONDIR=`pwd`/ipython 

alias gs="git status"
alias gb="git branch -v"
alias gcm="git commit -m"
alias gpullom="git pull origin master"
alias gpushom="git push origin master"



