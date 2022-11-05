# CEC-Ranking
Certified Error Control of Candidate Set Pruning for Two-Stage Relevance Ranking (EMNLP 2022 Oral).

## Installation
```
pip install -r requirement.txt
```
You also need to install [torch](https://pytorch.org/get-started/locally/) according to your configuration.
This project is also based on the [RCPS](https://github.com/aangelopoulos/rcps) library.

## Data
```
wget https://vault.cs.uwaterloo.ca/s/8H8gSLFMRSMdz8L/download -O msmarco.zip
unzip msmarco.zip

wget https://vault.cs.uwaterloo.ca/s/WBQ4F6DnHcjmc2g/download -O quora.zip
unzip quora.zip
```


## Calibration and Test
After downloading and decompressing the data, you could run the following code to calibrate and test your candidate set pruning method.
```
bash cec_prediction.sh
```

## License
<a href="https://opensource.org/licenses/MIT" alt="License">MIT License</a>
