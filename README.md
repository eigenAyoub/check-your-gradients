I'm starting a backprop project. Expect to see a lot of Gradients, Hessians, JAX and PyTorch.

See you!

I'll be working daily on this project.
 
### Abstract:
My main goal is to learn Pytorch/Jax, but also have a clear vision on what's happening underneath. I intend to build and train NNs from scratch, but have a PyTorch/Jax implelemention in parallel for correctness. Similar to what Andrej Karpathy have done in [Micrograd](https://github.com/karpathy/micrograd) with Pytorch.


### Goals:

* Be able to build and train NNs from scratch.
* Focus on Modularity, and Optimization of code.
* Learn & check correctness with PyTorch/Jax.
* See how it is done PyTorch/Jax on a deeper level.

### Steps:
1. **[IN-PROGRESS]** Build a starting NN:
    * **[DONE]** Set up a working env:
        * Load MNIST
    * **[DONE]** Train a simple MLP to classify MNIST
        * Forward.
        * Backward.
        * Loop and make sure the NN is learning.
        * **[DONE]** Current accuracy around 94%.
            * Config: simple MLP, RELU, SIGMOID and quadrqatic loss.
    * **[TO-DO]** Checkings:
        * Check why the Cross-Entropy layer is performing (weirdly) worse.
        * See how you can replicate your settings with PyTorch/Jax.
2. **[TO_DO]** Optimize:
    * Learn how to load the batches faster, using Iterators.
    * Improve your printing ("Cool people call it debugging")
    * Vertorize the high-dim tensors manipulations (the softmax layer)
3. [TO-DO] Make it modular.
    * Gets some inspirations from [this](https://www.youtube.com/playlist?list=PLeDtc0GP5ICldMkRg-DkhpFX1rRBNHTCs), Pytorch and Grokking DL.
4. [TO-DO] Start the project.


### References:
* Andrej Karpathy
* Grokking DL

