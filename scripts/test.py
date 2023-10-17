#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

test_generator.reset()  
y_pred = model.predict(test_generator)
y_pred_binary = np.round(y_pred)

y_true = test_generator.classes

target_names = list(test_generator.class_indices.keys())
class_report = classification_report(y_true, y_pred_binary, target_names=target_names)

conf_matrix = confusion_matrix(y_true, y_pred_binary)

print(class_report)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('predicted')
plt.ylabel('real')
plt.title('confusion matrix')
plt.show()

