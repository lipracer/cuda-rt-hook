#!/bin/bash

set -e

# git submodule update --init --recursive

set -e -u -x

# if command -v apt &> /dev/null; then
#     apt install -y cmake
# elif command -v yum &> /dev/null; then
#     yum install -y cmake
# else
#     echo "unknown os!"
# fi

# if [ -d "/io/dist" ]; then
#     rm -rf /io/dist    
# fi
mkdir -p /io/dist

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/dist/
    fi
}

# Compile wheels
for PYBIN in /opt/python/*/bin; do

    INCLUDE_DIR=`${PYBIN}/python -c "import sysconfig; print(sysconfig.get_path('include'))"`
    LIBRARY_DIR=`${PYBIN}/python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"`

    export PYTHON_INCLUDE_DIR=$INCLUDE_DIR
    export PYTHON_LIBRARY=$LIBRARY_DIR
    export PUBLISH_BUILD=1

    rm -rf /io/build
    "${PYBIN}/pip" install -r /io/requirements.txt
    # "${PYBIN}/pip" wheel /io/ --no-deps -w dist/
    "${PYBIN}/python" /io/setup.py sdist bdist_wheel --dist-dir=/io/tmp_dist
done

# Bundle external shared libraries into the wheels
for whl in /io/tmp_dist/*.whl; do
    repair_wheel "$whl"
done

rm -rf /io/tmp_dist

# python setup.py sdist bdist_wheel

# new_name=`ls dist | grep whl | awk -F'-' '{print $1"-"$2"-py3-none-any.whl"}'`

# ls dist | grep whl | xargs -I {} mv dist/{} dist/$new_name
