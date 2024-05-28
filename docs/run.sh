#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <notebook_name|--all>"
    exit 1
fi

# Execute notebook inplace, remove metadata
COMMAND="jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to notebook --inplace --execute"

# Run all Jupyter notebooks in the current directory
if [ "$1" == "--all" ]; then
    for notebook in *.ipynb; do
        $COMMAND "$notebook" 
        if [ $? -eq 0 ]; then
            echo " --> $notebook executed successfully"
        else
            echo "!!!!! Error executing $notebook !!!!!"
        fi
    done
else
    $COMMAND "$1"
    if [ $? -eq 0 ]; then
        echo " --> $1 executed successfully"
    else
        echo "!!!!! Error executing $1 !!!!!"
    fi
fi

echo "All notebooks have been run"
