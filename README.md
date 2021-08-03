# Application of the rmckenna mechanism to the adult dataset

This repository is based on the mechanism used by
[Ryan McKenna](https://people.cs.umass.edu/~rmckenna) who won the
[first place](https://www.nist.gov/ctl/pscr/team-rmckenna) in the
[Differential Privacy Synthetic Data Challenge](https://www.nist.gov/ctl/pscr/open-innovation-prize-challenges/past-prize-challenges/2018-differential-privacy-synthetic)
of the NIST in 2018. It is a modification of
[his submitted solution](https://github.com/usnistgov/PrivacyEngCollabSpace/tree/master/tools/de-identification/Differential-Privacy-Synthetic-Data-Challenge-Algorithms/rmckenna)
to run on the
[adult dataset](https://archive.ics.uci.edu/ml/datasets/adult).



## Installation

Install the Python3 working environment.

```shell
# Create the virtual environment
python3 -m venv venv

# Enter in the virtual environment
. venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

Clone the [private-pgm repository](https://github.com/ryan112358/private-pgm)
in another directory and add it to the python path. You can also add the
configuration of the python path to your `.bashrc` to load it automatically.

```shell
# Clone it in another directory
cd ..

# Clone the private-pgm repository
git clone https://github.com/ryan112358/private-pgm

# Add the src directory to the python path
export PYTHONPATH=$PYTHONPATH:`pwd`/private-pgm/src
```



## Dataset

The adult dataset can be downloaded from
[this link](https://archive.ics.uci.edu/ml/datasets/adult). Afterwards,
format it using `notebooks/adult-preprocess.ipynb` and generate the required
domain information file using `notebooks/adult-domain.ipynb`.



## Generate a synthetic dataset

Use the following command to generate a synthetic dataset. You can also
configure the parameters (use `--help` to list them).

```shell
python adult.py  # --help displays the parameters
```



<!-- TODO ## Extension to another dataset -->



## Execution on the GPU

### Installation

Check that the driver of your graphics card is installed and that it supports
cuda. Install cuda from
[the website of Nvidia](https://developer.nvidia.com/cuda-downloads) and reboot
your computer after the installation.

Check your version of cuda using `/usr/local/cuda/bin/nvcc --version` or
`nvcc --version`. If your version of cuda is 11, install the corresponding
version of pytorch using:

```shell
# If your version of cuda is >= 11
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Check that torch has access to cuda:

```shell
python -c "import torch; print(torch.cuda.is_available())"
```

A little modification has to be done on the sources of the `private-pgm`
repository. Add the following lines before the line 267 that sets the `diff`
variable:

```python
# If we are using the torch backend, the Q linear operator (of
# type matrix.Identity) was not a Tensor and generated an
# error due to this unaccepted format for Tensor operations.
if all((self.backend == 'torch',
        str(type(Q)) == "<class 'matrix.Identity'>")):
    import torch

    # First, we transform Q into a numpy array
    q_as_numpy_array = Q * np.identity(Q.shape[1])

    # Then, we format this numpy array to a Tensor
    Q = torch.as_tensor(q_as_numpy_array, dtype=torch.float32,
                        device=self.Factor.device)

    # Just a verification that the formatting of Q keeps the
    # same values. It is the case for the multiple executions
    # that I launched, you can keep or remove this as you want.
    assert np.array_equal(q_as_numpy_array, Q.cpu().numpy())
```


### Usage

You can generate a synthetic dataset using GPU by setting the backend parameter
to `torch`.

```shell
python adult.py --backend torch  # use --help instead to display the parameters
```

You can monitor the usage of the GPU by `watch -d -n 0.5 nvidia-smi`. You can
also use nvtop (`sudo apt install -y nvtop` then `nvtop`).
