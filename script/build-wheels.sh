#!/bin/bash

set -e

git submodule update --init --recursive
pip install -r requirements.txt
python setup.py sdist bdist_wheel

new_name=`ls dist | grep whl | awk -F'-' '{print $1"-"$2".whl"}'`

ls dist | grep whl | xargs -I {} mv dist/{} dist/$new_name
