I'm starting an Autodiff project. Expect to see a lot of Gradients, Hessians, JAX and PyTorch. I'll be regularly updating. The progress of the project is intentially left as the last section of this page. See you!

## ToC:
1. Abstract.
2. Goals.
3. References.
4. Progress.


### Abstract:
My main goal is to learn Pytorch/Jax, but also have a clear vision on what's happening underneath. I intend to build and train NNs from scratch, but have a PyTorch/Jax implelemention in parallel for correctness. Similar to what Andrej Karpathy have done in [Micrograd](https://github.com/karpathy/micrograd) with Pytorch.


### Goals:
* Be able to build and train NNs from scratch.
* Focus on Modularity, and Optimization of code.
* Learn & check correctness with PyTorch/Jax.
* See how it is done in PyTorch/Jax on a deeper level.

### References:
* Andrej Karpathy: [YouTube](https://www.youtube.com/@AndrejKarpathy) or [CS231n 2016](http://cs231n.stanford.edu/2016/)
* Mathieu Blondel: github repo and Section in new version of Murray Book
* Bishop NN Chapter has a section on backprop
* Hennig (Lecture 8, Learning Representations)
* TinyGrad by George Hotz

### Progress:
4. **[TO-DO]** Start the project.
    1. (Live)Current Plan:
        * Using **graphviz**, build a visualizer of the Computation Graph (Forward Pass).
            * [Check this Guide](https://www.graphviz.org/pdf/dotguide.pdf)
        * Build Autodiff on Simple MLPs: 
            * Make this NN work: $L = \sigma(W_2 ReLU(X W_1 + b_1) + b_2)$ 
        * Keep updating the plan (Include, Regul? Batch Norm? Fancy Optimizers?) 
3. **[TO-DO]** Make it modular.
    * Gets some inspirations from [this](https://www.youtube.com/playlist?list=PLeDtc0GP5ICldMkRg-DkhpFX1rRBNHTCs), Pytorch and Grokking DL.
2. **[TO-DO]** Optimize:
    * Learn how to load mini-batches faster, perhaps using Iterators?.
    * Improve your printing ("Cool people call it debugging skills")
    * Vertorize the high-dim tensors manipulations (the softmax layer)
1. **[IN-PROGRESS]** Build a starting NN:
    * **[DONE]** Set up a working env:
        * Load MNIST
    * **[DONE]** Train a simple MLP to classify MNIST
        * Forward.
        * Backward.
        * Loop and make sure the NN is learning.
        * Current accuracy around 94%.
            * Config: simple MLP, RELU, SIGMOID and quadrqatic loss.
    * **[TO-DO]** Checks:
        * Check why the Cross-Entropy layer is performing (weirdly) worse.
        * See how you can replicate your settings with PyTorch/Jax.

