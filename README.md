# Domain_oriented_Logic_Coverage

Experiment code associated with our paper:
AMANDA: A Memetic Algorithm for Domain-oriented MC/DC coverage

AMANDA is a domain oriented test adequacy for logic coverage criteria. In this approach, by searching the region of the program under test and selecting an instance from each sub-domain, by adding them to the test set that still statisfy the MC/DC criterion, we reach a new criterion called domain-oriented MC/DC.

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
