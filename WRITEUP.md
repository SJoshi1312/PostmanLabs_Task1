So basically first I just built the forward channel.
We went from 784 neurons in INPUT LAYER --> 128 neurons in 1st HIDDEN LAYER --> 64 neurons in 2nd HIDDEN LAYER --> 10 neurons in OUTPUT LAYER
Weights and biases were randomized in the beginning.
Ofcourse, as expected on the first run, without any backprop, the output neurons were about 0.1 for all the 10 digits, which corresponds to guessing. 

Afterwards, I implemented backpropagation using ReLU and not sigmoid. I created a loss function, along with a cross entropy function, which allowed me to tell how much the weights and biases were supposed to change. Ofcourse I generalised these functions so I could call them once from [OUTPUT LAYER to 2nd HIDDEN LAYER], from [2nd HIDDEN LAYER to 1st HIDDEN LAYER], and from [1st HIDDEN LAYER  to INPUT LAYER].

After implementing the backprop, accuracy bumped up to around 0.74-0.75.

I implemented a gradient decent algorithm after this to save the changes in weights and biases as they were occuring. 
This also bumped up the accuracy to about 0.892, and then eventually to 0.899

Finally, I modified the code slightly to process 32 images as a vector simultaneously instead of going 1 image at a time, concluding to process 5 batches of 32 images each with accuracy of 0.899 
