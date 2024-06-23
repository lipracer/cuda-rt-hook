#!/bin/bash

HEAD=`git log --oneline -n 2 | head -n 1 | awk '{print $1}'`
PRED_HEAD=`git log --oneline -n 2 | tail -n 1 | awk '{print $1}'`

echo $HEAD
echo $PRED_HEAD

git status

git log --stat -10 | cat

git clang-format $PRED_HEAD

CHANG_FILES=`git status --ignore-submodules --column --short | awk '{print $2}'`
echo $CHANG_FILES

for filename in $CHANG_FILES; do
    python -c "assert not '$filename'.endswith('.h'),'$filename'"
    python -c "assert not '$filename'.endswith('.hpp'),'$filename'"
    python -c "assert not '$filename'.endswith('.cc'),'$filename'"
    python -c "assert not '$filename'.endswith('.cpp'),'$filename'"
done