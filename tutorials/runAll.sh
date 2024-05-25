#!/bin/bash

# Run all Jupyter notebooks in the current directory and execute them in-place
for notebook in *.ipynb; do
    echo "Running $notebook ..."
    jupyter nbconvert --to notebook --execute "$notebook" --inplace
    if [ $? -eq 0 ]; then
        echo " --> $notebook executed successfully"
    else
        echo "!!!!! Error executing $notebook !!!!!"
    fi
done

echo "All notebooks have been run"
