# Mixture of Experts: When Does It Really Deliver Energy Efficiency?

One question I've been curious about is whether the mixture of experts (MoE)
architecture is more energy efficient.  I've seen quite a few claims that
it is but I haven't ever seen it measured or proven out. 

Intuitively, it makes some sense that if I only have a subset of a model's parameters
active then I should likely need fewer total operations on the model during
inference to produce an output.  But all this comes with a more complex 
architecture and some of those layers don't come for free. 
So I decided to test it on simple models to get a good comparison and
it turns out that the efficiency gains are conditional on model size and 
number of experts activated.  So MoEs are a general solution for improved 
efficiency only when certain properties on model size and expert 
activation are achieved.

## Model Design 
First off let's look at our experiment design.  I've provided the code here: 
https://github.com/neuralwatt/moe_energy_evaluation.  There are three models
included in this repo.
1. baseline_mlp.py
2. moe_original.py
3. moe_optimized2.py 

The first model is just a very basic multi-layer perceptron (MLP) where it's easy for me to configure the total 
number of parameters.  It's super basic just connecting pairs of Linear/ReLU layers
until we obtain the target parameter size.

The second model is a basic/naive Mixture of Experts implementation where every
expert is itself a basic MLP but where k experts are selected and outputs of 
the selected experts are combined using the learned weighting.

I also provide a third model which is an optimized version which reduces the number of 
iterations, improves input and output processing, and generally speeds up the forward call.

## Initial Test
Let’s compare these with a small set of parameters to test them.
###
Small params:
"NUM_EXPERTS": 8,
"EXPERT_HIDDEN_DIM": 256,
"EXPERT_NUM_LAYERS": 3,
"TOP_K_EXPERTS": 2,

Using the Small config in the MoE.py driver script our MoE model gets initialized with 2,687,192
parameters.  We use this to configure our baseline MLP model with a very close number of parameters:  
2,687,737 (getting identical would be an exercise in frustration since these architectures are different
enough and unnecessary since we are really talking about broad strokes in this article). And we proceed 
to train all the models for 15 iterations against a synthetic, but learnable, target.

### When MoE Fails: The Surprising Cost of Small Models
Energy Efficiency Results (Inferences per Joule):
- Original MoE: 742.50 inferences/Joule
- Optimized MoE: 770.18 inferences/Joule
- Baseline: 5746.08 inferences/Joule
We see here that both our MoE models are around 14% as efficient as our baseline model in terms of 
inferences per joule (the test driver runs inference in a loop for 30 seconds while summing the energy).
So, we see our first indication that making MoE more efficient isn't as simple as it seems.

The core issue is that even though we are running with identical trained parameters (and many fewer activated
parameters in the MoE model) the inference at this scale is much slower for MoE due to the computation complexity
of selecting the correct model and averaging the outputs compared with the very simple baseline.

So, what we really need to have is enough computational gains in our experts as they scale up to compensate 
for the slower model selection/merge code.  Let's try something bigger.

Here is our next config 
"NUM_EXPERTS": 16,
"EXPERT_HIDDEN_DIM": 1024,
"EXPERT_NUM_LAYERS": 4,
"TOP_K_EXPERTS": 4,
Which results in 63,418,800 trainable parameters (baseline gets 63,426,943).
Energy Efficiency Results (Inferences per Joule):
- Original MoE: 257.15 inferences/Joule
- Optimized MoE: 265.32 inferences/Joule
- Baseline: 337.49 inferences/Joule

We're getting close to overcoming the expert overhead but still aren't there.
Let's go bigger.

### The Turning Point: Where MoE Overtakes the Baseline 
"NUM_EXPERTS": 20,
"EXPERT_HIDDEN_DIM": 2048,
"EXPERT_NUM_LAYERS": 5,
"TOP_K_EXPERTS": 5,
Which results in 368,287,260 trainable parameters (baseline gets 368,360,149).

Energy Efficiency Results (Inferences per Joule):
- Original MoE: 89.91 inferences/Joule
- Optimized MoE: 95.35 inferences/Joule
- Baseline: 57.37 inferences/Joule

We managed to overcome the model selection and final weighting overhead. Now our MoE model is about
66% more energy efficient than our similar-sized MLP model. 

### But wait there is a surprise.
It turns out that our baseline model is still quite a bit faster than our MoE model.
Generating 17,151 inferences per second versus our optimized MoE model with 
15,762 inferences per second.  But the power use is much higher for our MLP hitting
max TDP (300W) for this A100 80GB for most of the inference test while the MoE model is just 
over 50% of TDP for its average power.  So, we are seeing the impact of the computational 
density on the number of operations (which directly impacts the power) and which leads
to our energy efficiency shifting in favor of the MoE even with its slower inference rate.

--- Optimized MoE Inference Timing ---
Actual Duration: 30.0053 seconds
Total Inferences: 472960
Inferences per Second: 15762.55
Total Energy Consumed: 4960.23 Joules
Average Power: 165.31 Watts

--- Baseline Inference Timing ---
Actual Duration: 30.0005 seconds
Total Inferences: 514560
Inferences per Second: 17151.69
Total Energy Consumed: 8969.24 Joules
Average Power: 298.97 Watts

This power pattern is generally demonstrated across all three sized models where the MLP 
model is generally drawing around 2x the power as the MoE.  Now I have a new question.
Since our MLP model is now hitting our GPU TDP can we double the size of the models and 
see some new properties emerge since at that point we should force close to full TDP of 300 W 
for both models.

### XXL 
"NUM_EXPERTS": 32,
"EXPERT_HIDDEN_DIM": 4096,
"EXPERT_NUM_LAYERS": 6,
"TOP_K_EXPERTS": 8,
We get 2,789,237,600 trainable parameters in our MoE model (2,789,253,815 for Baseline).  This is still
very small in the grand scheme of state-of-the-art MoE models (here is a good summary from the authors
of the MoE optimization paper referenced later https://github.com/MoE-Inf/awesome-moe-inference/) but 
big enough that we can almost saturate the single GPU compute we are using.

--- Optimized MoE Inference Timing ---
Actual Duration: 30.0126 seconds
Total Inferences: 122496
Inferences per Second: 4081.49
Total Energy Consumed: 7564.89 Joules
Average Power: 252.06 Watts

--- Baseline Inference Timing ---
Actual Duration: 30.0044 seconds
Total Inferences: 81024
Inferences per Second: 2700.41
Total Energy Consumed: 8741.06 Joules
Average Power: 291.33 Watts

Or 
Optimized MoE: 16.19 inferences/Joule
Baseline: 9.27 inferences/Joule

So, we are still maintaining the increasing efficiency rates for the MoE model while
finally overcoming the speed of the baseline as well.

### Other Details
I've tried to keep this pretty simple and restricted to some basic measurements.
There are certainly other techniques used with more advanced MoE architectures, optimizations,
distributing layers to other gpus/cpus, mixed precision which I haven't bothered
with but which do impact the energy efficiency.  I also limited the measurements
in these tests to GPU power since those are the easiest to get but by doing the 
inference in timebound blocks the rest of the system power should be roughly 
equivalent between tests in this setup.

There is at least one paper which calls out the issue with the computational overhead
added by the input/output layers.  

"The energy consumption patterns of MoE models present unique challenges due to their sparse activation patterns
 and distributed nature. While sparse activation theoretically reduces computational demands, the overhead from expert
 routing, communication, and load balancing can lead to significant energy costs [68, 123, 162]. Current hardware
 platforms, often optimized for dense computations, result in suboptimal energy efficiency when handling the dynamic
 workloads characteristic of MoE inference [125, 231]. Additionally, the distributed nature of many MoE deployments
 introduces substantial energy overhead from data movement and communication."[1]

### Call to Action
I've made the code available here: https://github.com/neuralwatt/moe_energy_evaluation
Feel free to try it, extend it with other experiments, or provide feedback to me on
this.

We see about half of the SoA LLMs using MoE today and it's not yet clear whether that 
will continue to grow or whether a new architecture will emerge that better balances
parameter activation and model quality.  

At Neuralwatt we're working hard to make AI more efficient.  Feel free to reach out
to scott@neuralwatt.com if you are struggling with power limits, LLM performance, or 
measure the CO2 impact of your AI.  We're here.

### System Config
Single Nvidia A100 80G
Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz 12 vcpus
Crusoe Cloud

[1] https://arxiv.org/pdf/2412.14219 JIACHENG LIU, PENG TANG et el. A Survey on Inference Optimization Techniques for Mixture of Experts Models
![image](https://github.com/user-attachments/assets/a5117d0d-c745-4e7b-ab35-ca7c51b822c3)
