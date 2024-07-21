# Lesson0: Introduction

**Key Takeaways:**

`Prompting` applies not just to **text** but also to **vision**, including some image segmentation, object detection and image generation models.

Depending on the `vision model`, the prompts may be **text**,
but it could also be **pixel coordinates** or **bounding boxes** or **segmentation**.

In this course, you'll prompt [Meta Segment Anything Model (SAM)](https://segment-anything.com/) to identify the outline of an object by giving it some coordinates or points or bounding boxes to help it identify the object.
- For example, maybe a t-shirt that the dog is wearing. 

You also apply `negative prompts` to tell the model which regions to exclude when identifying an object.

Using a combination of positive and negative prompts, helps you to isolate a region the object that you are interested in, such as maybe a specific pattern on a t-shirt that a dog is wearing.

You'll learn about many other tools, as well as best practices for prompting to:
- Analyze status to understand images, as well as to generate or to otherwise manipulate or change images.
  
In the course, you'll also apply prompt engineering to image generation: 
- For instance, you can provide the text prompt a dragon to the **stability diffusion** model to generate:

```
the image of a dragon
```

and you'll `iterate` on that prompt. For instance, 

```
a realistic green dragon.
```

To get a different image of a dragon, you will also prompt the **diffusion model** to

```
replace a photograph of a cat with the dragon, while keeping the rest of the photograph intact.
```

> This is called `inpainting`, in which you edit a painting or photograph by removing a segmented object and replacing it with a generated image.

For inpainting, your prompt will not only be the text a realistic green dragon, but also an outline of the cat that the dragon will replace.

You'll obtain that outline or mask using image segmentation.

Furthermore, you'll obtain the bounding box that SAM uses as input by prompting an **object detection model**, this time with a text prompt such:

```
as cute dog with a pink jacket to generate a bounding box around that dog.
```

You will `iterate` on both `the prompts` and `the model hyperparameters` that you will tune in this inpainting pipeline.

> Diffusion models work by transforming a sample from a simple distribution like Gaussian noise, into a complex, learned distribution, like images. 

- The `guidance scale` hyperparameter determines 
how heavily the text input should affect the target distribution during the `reverse diffusion process`. 
- A `lower guidance scale` will allow the model to sample freely from its learned distribution of images
- A higher guidance scale will guide the model towards a sample that more closely matches the text input. 

The number of inference steps hyperparameter, controls how gradually the model transforms the noisy distribution back into a clean sample.

More steps generally allows for a more detailed and accurate generation, as the model has more opportunities to refine the data.

However, more steps also means more computation time.

More step generally allows for a more detailed and accurate generation, as the model has more opportunities to refine the data.

However, more steps also means more computation time.

The strength, hyperparameter, and the context of stable diffusion, determines how noisy the initial distribution is. 

During the inpainting process, where the added noise is used to erase portions of the initial image, strength essentially determines how much of the initial image is retained in the diffusion process.

Furthermore, what if you wanted to personalize
the diftusion model to generate not just a generic dragon, cat or person, but a specific dragon, your specific pet cat, or your best friend?

You'll use a fine tuning method called Dreambooth, developed by Google Research, to tune the stable diffusion model to associate a text label with a particular object, such as your good friend.

And of course, youll use the dreamboot tuning process on the stable diffusion model to associate the word Andrew Ng with just six photographs of Andrew.

After fine tuning, you can prompt the model with texts
such as a Van Gogh painting of Andrew, and the model can generate. the image.

One unique aspect of vision model development workflows
is that evaluation metrics won't always be able to tel you the ful story.

Oftentimes, you'll want to visualize your image outputs and inspect manually to understand where your model is
getting things right and where it's getting things wrong.

is that evaluation metrics won't always be able to tell you the full story. 

Oftentimes, you'll want to visualize your image outputs and inspect manually to understand where your model is getting things right and where it's getting things wrong.

And we'll talk through best practices for efficiently carrying out this type of iteration as well.

Let's say your object detection model is performing poorly all of a sudden, and your input data distribution hasn't changed.

You open up a few incorrect predictions and realize that a new object has been introduced to your images that your model is mistaking for your target object.

It's time to further train your model on this new object.

It's unlikely that evaluation metrics alone would it be able to paint the full story of what was going on here.

And so visualizing your output can be very important.

Similarly, when iterating across different sets of hyperparameters,

you'll sometimes need to see the output image in order to understand how the hyperparameter values are affecting it.

Experiment tracking tools can help you compare these output
images side by side and track and organize, which inputs lead to which outputs so you can reproduce them later on.

Computer vision workflows are highly iterative, so it's valuable to track each of your experiment runs.

Many people have worked to create this course.


And so visualizing your output can be very important.

Similarly, when iterating across different sets of hyperparameters, you'll sometimes need to see the output image in order to understand how the hyperparameter values are affecting it.

Experiment tracking tools can help you compare these output images side by side and track and organize, which inputs lead to which outputs so you can reproduce them later on.

Computer vision workflows are highly iterative, so it's valuable to track each of your experiment runs.

Many people have worked to create this course.

I'd like to thank on the Comet's side, Sid Mehta, senior growth engineer at Comet.

From DeepLearning.AI Eddy Shyu also contributed to this course.

In the first lesson, you'll get an overview of visual prompting for image segmentation, object detection, and diffusion models that you'll use in this course.

> I think you could be a real visionary when it comes to prompting.

## References

- Main Course: https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction

Diffusion Models Resources:

- Docs, notes: 
  - [Diffusion Models Notes - @afondiel](https://github.com/afondiel/computer-science-notes/blob/master/ai/generative-ai-notes/Diffusion-notes/diffusion-models-notes.md)
  - [Vision Large Models (VLMs) Notes - @afondiel](https://github.com/afondiel/computer-science-notes/tree/master/computer-vision-notes/vision-large-models-VLMs)

- HF Diffusion models tools: 
  - [Diffusion Course - HF learn](https://huggingface.co/learn/diffusion-course/unit0/1)
  - [HF Diffusers - library & pipeline for SOTA pretrained diffusion models](https://huggingface.co/docs/diffusers/index) 
