# LPCV 2023 - Tiny Espresso Net

## Compressing to pyz

From outside solution directory in which solution is the directory name: `python3.6 -m zipapp  solution  -p='/usr/bin/env python3.6'`

## Formatting

This is the directory tree from our sample solution and in correspondence with the path we used for our model.

```
solution
├── __init__.py
├── __main__.py
├── main.py
├── model.pkl
├── README.md
└── utils
    ├── fanet.py
    ├── __init__.py
    ├── README.md
    └── resnet.py
```

main.py has a path to 'model.pkl' make sure to update it with the name of your model and make sure your model is in the solution directory.

## Evaluation

We will be evaluating your file by using the evaluation folder provided.

## Run Solution

The Solution should be able to be ran with the command: `python3.6 solution.pyz -i /path/to/imageDirectory -o /path/to/outputDirectory`.
If running with the Jetson Nano and you are getting "illegal instruction (core Dumped) try running `export OPENBLAS_CORETYPE=ARMV8`.
