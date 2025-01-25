# 2. Classification Using CNN 

## 2.3 Hyperparameter tuning 

### Following are the 5 combinations:

hyperparameter_combinations = [
    {'learning_rate': 0.01, 'dropout_rate': 0.5, 'num_conv_layers': 3, 'optimizer': 'adam', 'task': 'classification'},
    {'learning_rate': 0.005, 'dropout_rate': 0.3, 'num_conv_layers': 3, 'optimizer': 'adam', 'task': 'regression'},
    {'learning_rate': 0.01, 'dropout_rate': 0.5, 'num_conv_layers': 2, 'optimizer': 'sgd', 'task': 'classification'},
    {'learning_rate': 0.005, 'dropout_rate': 0.3, 'num_conv_layers': 4, 'optimizer': 'rmsprop', 'task': 'regression'},
    {'learning_rate': 0.01, 'dropout_rate': 0.5, 'num_conv_layers': 3, 'optimizer': 'adam', 'task': 'classification'},
]


Epoch [1/10], Training Loss: 1.3474, Validation Loss: 1.1018
Epoch [2/10], Training Loss: 1.1440, Validation Loss: 1.1037
Epoch [3/10], Training Loss: 1.1439, Validation Loss: 1.1046
Epoch [4/10], Training Loss: 1.1427, Validation Loss: 1.1072
Epoch [5/10], Training Loss: 1.1420, Validation Loss: 1.1047
Epoch [6/10], Training Loss: 1.1421, Validation Loss: 1.1041
Epoch [7/10], Training Loss: 1.1420, Validation Loss: 1.1031
Epoch [8/10], Training Loss: 1.1418, Validation Loss: 1.1034
Epoch [9/10], Training Loss: 1.1420, Validation Loss: 1.1053
Epoch [10/10], Training Loss: 1.1419, Validation Loss: 1.1043

Hyperparameter Combination 1:
Training Loss: 1.1419
Validation Loss: 1.1043, Validation Accuracy: 0.5333
Best model saved!

Epoch [1/10], Training Loss: 2.5253, Validation Loss: 0.1389
Epoch [2/10], Training Loss: 0.1866, Validation Loss: 0.0608
Epoch [3/10], Training Loss: 0.1536, Validation Loss: 0.0303
Epoch [4/10], Training Loss: 0.1908, Validation Loss: 0.0948
Epoch [5/10], Training Loss: 0.1054, Validation Loss: 0.0172
Epoch [6/10], Training Loss: 0.0746, Validation Loss: 0.0138
Epoch [7/10], Training Loss: 0.0580, Validation Loss: 0.0118
Epoch [8/10], Training Loss: 0.0455, Validation Loss: 0.0108
Epoch [9/10], Training Loss: 0.0419, Validation Loss: 0.0119
Epoch [10/10], Training Loss: 0.0381, Validation Loss: 0.0087

Hyperparameter Combination 2:
Training Loss: 0.0381
Validation Loss: 0.0087, Validation Accuracy: 0.9893
Best model saved!

Epoch [1/10], Training Loss: 0.5477, Validation Loss: 0.0838
Epoch [2/10], Training Loss: 0.1287, Validation Loss: 0.0077
Epoch [3/10], Training Loss: 0.0659, Validation Loss: 0.0036
Epoch [4/10], Training Loss: 0.0271, Validation Loss: 0.0007
Epoch [5/10], Training Loss: 0.0282, Validation Loss: 0.0024
Epoch [6/10], Training Loss: 0.0306, Validation Loss: 0.0007
Epoch [7/10], Training Loss: 0.0213, Validation Loss: 0.0001
Epoch [8/10], Training Loss: 0.0251, Validation Loss: 0.0008
Epoch [9/10], Training Loss: 0.0162, Validation Loss: 0.0002
Epoch [10/10], Training Loss: 0.0142, Validation Loss: 0.0038

Hyperparameter Combination 3:
Training Loss: 0.0142
Validation Loss: 0.0038, Validation Accuracy: 0.9987
Best model saved!

Epoch [1/10], Training Loss: 1109594590.9948, Validation Loss: 0.0598
Epoch [2/10], Training Loss: 0.2680, Validation Loss: 0.5413
Epoch [3/10], Training Loss: 0.1763, Validation Loss: 0.1028
Epoch [4/10], Training Loss: 0.1436, Validation Loss: 0.1423
Epoch [5/10], Training Loss: 0.1771, Validation Loss: 0.0761
Epoch [6/10], Training Loss: 0.1083, Validation Loss: 0.0155
Epoch [7/10], Training Loss: 0.0964, Validation Loss: 0.0340
Epoch [8/10], Training Loss: 0.1054, Validation Loss: 0.0109
Epoch [9/10], Training Loss: 0.0477, Validation Loss: 0.0061
Epoch [10/10], Training Loss: 0.0331, Validation Loss: 0.0045

Hyperparameter Combination 4:
Training Loss: 0.0331
Validation Loss: 0.0045, Validation Accuracy: 0.9947

Epoch [1/10], Training Loss: 0.4086, Validation Loss: 0.0144
Epoch [2/10], Training Loss: 0.0680, Validation Loss: 0.0075
Epoch [3/10], Training Loss: 0.0295, Validation Loss: 0.0009
Epoch [4/10], Training Loss: 0.8435, Validation Loss: 1.9859
Epoch [5/10], Training Loss: 1.2056, Validation Loss: 1.1043
Epoch [6/10], Training Loss: 1.1425, Validation Loss: 1.1074
Epoch [7/10], Training Loss: 1.1416, Validation Loss: 1.1061
Epoch [8/10], Training Loss: 1.1418, Validation Loss: 1.1014
Epoch [9/10], Training Loss: 1.1423, Validation Loss: 1.1043
Epoch [10/10], Training Loss: 1.1421, Validation Loss: 1.1044

Hyperparameter Combination 5:
Training Loss: 1.1421
Validation Loss: 1.1044, Validation Accuracy: 0.5333



Best Hyperparameters:
{'learning_rate': 0.01, 'dropout_rate': 0.5, 'num_conv_layers': 2, 'optimizer': 'sgd', 'task': 'classification'}

Best Model - Validation Accuracy: 0.9987, Validation Loss: 0.0038
Best Model - Test Accuracy: 0.9993, Test Loss: 0.0023



