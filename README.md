# multi-inpaint-coco

Choosing the best evaluation prompt:

Prompt | method | acc |
--- | --- | --- |
Based on the two images, ordered from left to right, which follows the following caption, in the best way \"{caption}\"? Left or right? Answer in English. | exact_match | 50.54% |
Given the two images, ordered from left to right, and the caption \"{caption}\", does the caption describe better the image in the right? Yes or no? Answer in English. | exact_match | 50.93% (only "yes") |
