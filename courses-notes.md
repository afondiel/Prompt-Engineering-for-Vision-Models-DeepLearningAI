# Prompt Engineering for Vision Models Crash Course - Notes


## Lesson0: Introduction

**Key Takeaways:**

- Prompting applies not just to `text` but also to `vision` tasks: images segmentation, object detection and image generation models.
- In case of diffusion models, prompt can be: text, pixel coordinates or bounding boxes or segmentation
- Models used during the course: SAM (Segment Any Thing Model) which is used to identify the outlines of an object given some coordinates/points or bounding boxes to help it identify the object
- Negative prompts: tell the model which region to exclude when identifying an object
- Combining positive and negative prompt allow to isolate a region of the object that (you are interested) such as: a specific pattern in the image
- Many other tools and best pratices are covered during the course for image analysis, generation and manipulation.

Techniques:

Image generation (on diffusion models): 
- prompt creation and iteration
  - replace a specific segmented object in the the image by the generated image (inpainting)   

Inpainting:
- Prompting can be text, outline of the  of the image (image segmentation masks)

Object Detection: using bounding boxes 



## Lesson1: Overview
## Lesson2: Image Segmentation
## Lesson3: Object Detection
## Lesson4: Image Generation](#)
## Lesson5: Fine-tuning](#)
## Lesson6: Conclusion](#)




