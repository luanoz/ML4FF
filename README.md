# ML4FF

This repo contains the dataset and codes used in the paper “ML4FF: a benchmark for flash flood forecasting”.

# Project tree

 * [ML4FF.py](./ML4FF.py)
 * [Summary.xlsx](./Summary.xlsx)
 * [Summary_Perf.xlsx](./Summary_Perf.xlsx)
 * [data](./data)
 * [docs](./docs)
 * [Models](./Models)
 * [README.md](./README.md)

The ML4FF.py file contains the Python code developed to implement the ML4FF benchmark.

Summary.xlsx and Summary.xlsx are two Excel spreadsheets created using the functions build_excel and perf_excel, respectively, of the ML4FF.py file. The use of such functions is documented within the ML4FF.py file.

The data folder contains the dataset considered in the ML4FF benchmark.

The docs folder contains supplementary materials produced for the paper (extra figures)

The Models folder contains the pickled results of the benchmarks run for all the 32 algorithms. These files can be imported and manipulated, as documented within the ML4FF.py file.
