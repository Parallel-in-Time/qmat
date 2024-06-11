#!/bin/bash

echo "print('Loading sitecustomize.py...');import coverage;coverage.process_startup()" > sitecustomize.py
coverage run -m pytest --continue-on-collection-errors -v --durations=0 ./tests