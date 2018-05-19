Basic python loader for NYU datasets.

# Usage
1. Clone this repo and ensure the parent directory is on your python path.
```
cd /path/to/parent_dir
git clone https://github.com/jackd/nyu.git
```
2. Add the parent directory is on your `PYTHONPATH`
```
export PYTHONPATH=$PYTHONPATH:/path/to/parent_dir
```
3. Add the path to the NYU datasets to the NYU_PATH environment variable.
```
export NYU_PATH=/path/to/dataset/nyu
```
This should contain directories `v1` and `v2` with data/train-test split downloaded from [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v1.html) and [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) respectively.

# Examples
See `example` subdirectory.
