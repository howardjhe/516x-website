---
layout: default
---

# Introduction

Weed image classification is an important task in precision agriculture, which aims to improve farming practices by using advanced technologies and data-driven decision-making. The main objective of weed image classification is to distinguish between weed and crop plants in images taken from agricultural fields. By accurately identifying weed species and their locations, farmers can take targeted actions to control weed growth, leading to improved crop yields and reduced usage of herbicides. When it comes to binary classification, we can use supervised learning like Navie Bayes (NB), support vector machine (SVM) and convolutional neuron network (CNN). In order to implement to real-world tasks, inference time, computing cost and accuracy are important to be consider determining which the methods are most suitable to be used. Thus, in this project, we experimentally present which of the binary classification supervised learning methods (NB, SVM, or CNN) performs better in terms of accuracy and computing time for weed detection in soybean crops.

_All experiments are conducted in Python and run on a laptop with four 2.4 GHz cores and 16 GB of RAM._

# Workflow

To fulfill this project, we can follow the following steps:

- Data preparation:
  - Download the dataset from Kaggle [Link](https://www.kaggle.com/datasets/fpeccia/weed-detection-in-soybean-crops)
  - Rearrange the number of images for weed and non-weed into 260 and 5000, respectively.
  - Import the necessary packages.
  - Preprocess the data by resizing images, normalizing pixel values, and formatting the data type (for example, squeeze the dimension) in order to input the model. We use NumPy array as input for NB and SVM, and Tensor for CNN.
  - Split the dataset into training, and test sets with the sizes of 0.8 and 0.2, respectively.

```Python
# Import the necessary packages.
import numpy as np
import cv2
import os
import glob
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
```


Text can be **bold**, _italic_, or ~~strikethrough~~.

[Link to another page](./another-page.html).

There should be whitespace between paragraphs.

There should be whitespace between paragraphs. We recommend including a README, or a file with information about your project.

# Header 1

This is a normal paragraph following a header. GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.

## Header 2

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
