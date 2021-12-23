# Genetic Algorithms

### Multi-threaded execution
The following executable was needed in order to perform parameter tuning by
running multiple experiments in parallel and storing results in related numpy arrays.

Given the number of threads and the name of the problem (`one_max`, `leading_ones`, `labs`), the below script was used:
~~~
$ python genetic_algorithm_multithreading.py --threads <number-of-threads> --problem <problem_name>
~~~

### Single-threaded execution
After analyzing the results generated from the multi-threaded execution,
the following single-threaded execution was performed in order to get the data needed for IOH Analyzer which was used to analyze the results:
~~~
$ python genetic_algorithm_single.py 
~~~

### GA Framework (1 + (λ,λ))
In order to execute the alternative GA framework named `GA (1 + (λ,λ))` the below script can be used:
~~~
$ python alternative_ga.py   
~~~


### Collaborators
- [Andreas Paraskeva](https://www.linkedin.com/in/andreas-paraskeva-2053141a3/)
- [Pavlos Zakkas](https://www.linkedin.com/in/pzakkas/)