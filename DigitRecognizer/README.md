

| Model  | Public Leaderboard |
| ------------- | ------------- |
| Lenet-5  | ~0.992  |
| Content Cell  | Content Cell  |

## Lenet-5



### Fine tuning the performance for Lenet-5


<h4>Approach one:  Baby sitting the training</h4>

1. Training Lenet-5 model with random weights

model is under performing, since the training accuracy always lower than val accuracy
   

2. + dropouts
    This actually 

3. + learning rate decay


Conclusion


## Submit results:

```bash
kaggle competitions submit -c digit-recognizer -f submission.csv -m "Message"
```
