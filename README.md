# 340W Project, NPMM Labeling with Stable Rule Set Classifiers

This project needs Julia 1.9 to run. It also needs a valid Python distribution as it will install Python dependencies.

Package dependencies

Plots, LinearAlgebra, Statistics, DataFrames, CSV, NLsolve, Memoize, MLJBase, Pkg, MLJModels, ScikitLearn, Flux, Zygote, StatsBase, PyCall, Dates, MLJParticleSwarmOptimization, StatisticalMeasures, MLJTuning, Distributions, MethodChains, SIRUS, EvoTrees

Python Dependencies

ta, pandas, pandas_datareader, yfinance

Results are stored in the notebooks folder under airlines.ipynb, beverages.ipynb, and tech.ipynb. Data download commences on the first run, ensure that you have a stable internet connection. Startup and first recompilation times for Julia are long and Python dependencies will be downloaded by the script itself on the first launch to its own Anaconda environment. The Julia installations, Python installations, and data downloads can take more than 10 gigabytes of space, be aware of space constraints when downloading data. Unfortunately, julia's package manager at this time does not allow for sending a requirements.txt, Packages can be downloaded with these commands.

```
using Pkg
Pkg.add(split("Plots LinearAlgebra Statistics DataFrames CSV NLsolve Memoize MLJBase Pkg MLJModels ScikitLearn Flux Zygote StatsBase PyCall Dates MLJParticleSwarmOptimization StatisticalMeasures MLJTuning Distributions SIRUS EvoTrees", " "))
Pkg.add(url="https://github.com/uniment/MethodChains.jl")
```

