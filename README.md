# ThreeDP3

## Setup 
```
mkdir deps
cd deps
git clone git@github.com:probcomp/GenSceneGraphs.jl.git
git clone git@github.com:probcomp/GenDirectionalStats.jl.git
git clone git@github.com:probcomp/MeshCatViz.git
git clone git@github.com:probcomp/GLRenderer.jl.git
cd GLRenderer.jl/src/renderer
python setup.py develop
cd ../../../..
```
In Julia Package Manager:
```
dev ./deps/GenDirectionalStats.jl ./deps/GLRenderer.jl ./deps/MeshCatViz ./deps/GenSceneGraphs.jl
instantiate
```

To test that the setup has worked run:
```
julia test_generative_model.jl
```
