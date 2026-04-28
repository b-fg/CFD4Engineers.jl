## CFD4Engineers and ML @ TU Delft

Pluto programming notebooks for the ML lectures of the CFD4Engineers course. Contains

- [notebook1.jl](notebook1.jl): tutorial explaining how neural netoworks operate and comparing them to a constricted parametric function such as a paraboloid
- [notebook2.jl](notebook2.jl): tutorial building a finite volume solver for the Burgers equation from scratch, and deriving an optimal implicit subgrid-scale model using automatic differentiation


### Run the notebooks

To run a notebook, you need to [install Julia](https://github.com/JuliaLang/juliaup) and then [install Pluto.jl](https://plutojl.org/#install), the notebook tool to run the code interactively:

After installing Julia, open the Julia REPL (terminal), install Pluto, and run a Pluto server
```julia
import Pkg; Pkg.add("Pluto")
import Pluto; Pluto.run()
```
In the browser that Pluto opens, enter the notebook URL. For the first lecture:

```
https://github.com/b-fg/CFD4Engineers.jl/blob/main/notebook1.jl
```
And for the second lecture:
```
https://github.com/b-fg/CFD4Engineers.jl/blob/main/notebook2.jl
```

Then click on **"Run notebook code"**, and that's it! The notebook take about 20 minutes to download and install all the dependencies before running the code. Just grab a coffee and start reading it in the meantime ;)

#### Alternative: Clone this repo and run the local notebook
You can always just clone this repository, install Pluto following the same procedure as before, and enter the path to the local notebook.jl file.
