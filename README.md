# EvoDomain

Experiment code associated with our paper:
Improving Fault Detection and Localization Using a Search-Based Domain-Oriented Test Suite Generation

EvoDomain is a domain-oriented test adequacy for logic coverage criteria. In this approach, by searching the region of the program under test and selecting an instance from each sub-domain, by adding them to the test set that still satisfies the MC/DC criterion, we reach a new criterion called domain-oriented MC/DC.

## Architecture
![alt text](/AMANDA_diagram.jpg)

## 2D examples
Example 1                                        |  Example 2
:-----------------------------------------------:|:-----------------------------------------------:
![Alt text](/ex1.gif)                            |  ![Alt text](/ex2.gif)


## Required packages
- numpy
- matplotlib
- sklearn
- networkx
- graphviz
- ast
- astor
- truths
- pandas

## Usage
python3 /src/main.py

## Evaluation scripts
[Evaluation measurements for a domain-oriented test suite](/Evaluation%20measurements.ipynb)

[DBSCAN clustering to select test data from each subdomain](/Postprocessing.ipynb)

## Final result
To evaluate the EvoDomain approach, two categories of classic and industrial problems are defined. You can see more experimental details [here](https://www.dropbox.com/home/AMANDA).
