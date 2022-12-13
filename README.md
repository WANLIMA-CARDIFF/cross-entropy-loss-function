# cross-entropy-loss-function

This is a Pytorch implementation of cross entropy loss function, espectially for semantic segmentation / land cover classification, etc..

Usage:
```
hand_ce_loss = ESPL_CE(number_class)
yHat = model(x)
loss = hand_ce_loss(yHat, y)
```

There is a more detailed explanation in my [CSDN](https://blog.csdn.net/mawonly/article/details/128308722?spm=1001.2014.3001.5502) blog.
