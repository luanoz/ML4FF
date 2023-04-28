# ML4FF

This repo contains the dataset and codes used in the paper “ML4FF: a benchmark for flash flood forecasting” by Jaqueline A. J. P. Soares, Luan C. S. M. Ozelim,Luiz Bacelar, Dimas B. Ribeiro, Stephan Stephany and Leonardo B. L. Santos.

# Project tree

 * [ML4FF.py](./ML4FF.py)
 * [Summary.xlsx](./Summary.xlsx)
 * [Summary_Perf.xlsx](./Summary_Perf.xlsx)
 * [data](./data)
 * [docs](./docs)
 * [Models](./Models)
 * [figs](./figs)
 * [LICENSE](./LICENSE)
 * [README.md](./README.md)

The ML4FF.py file contains the Python code developed to implement the ML4FF benchmark.

Summary.xlsx and Summary_Perf.xlsx are two Excel spreadsheets created using the functions build_excel and perf_excel, respectively, of the ML4FF.py file. The use of such functions is documented within the ML4FF.py file.

The data folder contains the dataset considered in the ML4FF benchmark.

The docs folder contains the Readme file with general information on the code.

The Models folder contains the pickled results of the benchmarks run for all the 32 algorithms. These files can be imported and manipulated, as documented within the ML4FF.py file.

The figs folder contains PDF files of all the figures generated in this research, including those not presented in the paper due to space limitations.
