# ThreeDP3

## Setup 
```shell
git clone git@github.com:probcomp/ThreeDP3.git
cd ThreeDP3
julia --project -e 'import Pkg;
                    Pkg.pkg"dev --local git@github.com:probcomp/GenSceneGraphs.jl.git git@github.com:probcomp/GenDirectionalStats.jl.git git@github.com:probcomp/MeshCatViz.git git@github.com:probcomp/GLRenderer.jl.git";
                    Pkg.instantiate()'

python3 -m venv my_venv
source my_venv/bin/activate
cd dev/GLRenderer/src/renderer
python setup.py develop
cd ../../../..
```

To test that the setup has worked run:
```
julia --project notebooks/test_generative_model.jl
```
