#!/bin/bash

# Update path:
if [ -n "$PATH" ]; then
        PATH=`pwd`/mit_linux:`pwd`/fsmcontroller:`pwd`/clm:${PATH}
else
        PATH=`pwd`/mit_linux:`pwd`/fsmcontroller:`pwd`/clm
fi

# Update library path:
if [ -n "$LD_LIBRARY_PATH" ]; then
        LD_LIBRARY_PATH=`pwd`/predictivefFilter:`pwd`/centroid:${LD_LIBRARY_PATH}
else
	LD_LIBRARY_PATH=`pwd`/predictivefFilter:`pwd`/centroid
fi

# Ipython Notebook Set-up:
export IPYTHONDIR=`pwd`/ipython 

alias gs="git status"
alias gb="git branch -v"
alias gcm="git commit -m"
alias gpullom="git pull origin master"
alias gpushom="git push origin master"



