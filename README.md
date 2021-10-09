# ThreeDP3

## Setup 
```shell
# Install NVidia CUDA toolkit.  On Ubuntu:
sudo apt install nvidia-cuda-toolkit

# Download the Julia package and its dependencies
git clone git@github.com:probcomp/ThreeDP3.git
cd ThreeDP3
julia --project -e 'import Pkg;
                    Pkg.pkg"dev --local git@github.com:probcomp/GenSceneGraphs.jl.git git@github.com:probcomp/GenDirectionalStats.jl.git git@github.com:probcomp/MeshCatViz.git git@github.com:probcomp/GLRenderer.jl.git";
                    Pkg.instantiate()'

# Create a Python virtualenv to be used with the project
python3 -m venv my_venv
source my_venv/bin/activate
pip install --upgrade pip setuptools
# Rebuild Conda.jl and PyCall so that they link the new Python env
PYTHON=$(which python) PYCALL_JL_RUNTIME_PYTHON=$(which python) julia --project -e 'import Pkg; Pkg.build("Conda"); Pkg.build("PyCall")'

# Install the Python package
cd dev/GLRenderer/src/renderer
python setup.py develop
cd ../../../..
```

To test that the setup has worked run:
```
julia --project notebooks/test_generative_model.jl
```
