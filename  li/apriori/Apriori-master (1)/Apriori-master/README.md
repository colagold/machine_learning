# Apriori algorithm

## Overview
Apriori algorithm  is used to find relevant association rules over transactions dataset.

## Input
The input file is csv-file contained transactions. 

Outlook|Temperature|Humidity|Windy |PlayTennis
-------|-----------|--------|-------|----------
sunny  |hot        |high    |FALSE |N
sunny|hot|high|TRUE|N
overcast|hot|high|FALSE|P
rain|mild|high|FALSE|P
rain|cool|normal|FALSE|P
rain|cool|normal|TRUE|N
overcast|cool|normal|TRUE|P
sunny|mild|high|FALSE|N
sunny|cool|normal|FALSE|P
rain|mild|normal|FALSE|P
sunny|mild|normal|TRUE|P
overcast|mild|high|TRUE|P
overcast|hot|normal|FALSE|P
rain|mild|high|TRUE|N

## Output
The output file contains _n_-term sets and _2_-set rules.
```
Support=0.25
Confidence=0.25
1-term set:
[('Temperature', 'cool')]
[('Outlook', 'overcast')]
...
2-term set:
[('PlayTennis', 'P'), ('Outlook', 'overcast')]
[('PlayTennis', 'P'), ('Windy', 'FALSE')]
...
3-term set:
[('PlayTennis', 'P'), ('Humidity', 'normal'), ('Windy', 'FALSE')]
...
2-set rules:
Rule#1: {Humidity=normal} => {Temperature=cool}(Support=0.29, Confidence=0.57)
Rule#2: {PlayTennis=P} => {Windy=FALSE}(Support=0.43, Confidence=0.67)
Rule#3: {Humidity=high} => {Windy=FALSE}(Support=0.29, Confidence=0.57)
...
```

## Build and run
To run use (Python3 needed)
```
./RuleMining.py -s <min_support> -c <min_confidence> -i <input_file.csv> -o <output_file.txt>
```
For instance,
```
./RuleMining.py -s 0.25 -c 0.40 -i Play_Tennis_Data_Set.csv -o Rules_out.txt
```

## License
MIT
