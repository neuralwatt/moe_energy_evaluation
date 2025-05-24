# Energy Efficiency of State of the Art Open Weight Mixture of Expert Models.

## This is the second post in a series about evaluating the energy efficiency of mixture of experts models (MoE).
The first post [here](https://www.neuralwatt.com/blog/mixture-of-experts-when-does-it-really-deliver-energy-efficiency)
examined at which point the overhead to select and average experts overcame the efficiencies of having experts. 
This raised the question of how these toy models compared against current state of the art dense and MoE models.

### Framing the problem.
In part 1 I creted toy models to be able to directly compare number of tunable parameters.  Among the current available
open weight models it turns out that there isn't really great comparisons available which have close to same number
of tunable parameters.   Instead I decided to just start by comparing 
[Llama 3.3 70b (Q4_K_M quantization)](https://ollama.com/library/llama3.3:70b) dense model with the 
[Llama 4 scout 109B (Q4_K_M quantization)](https://ollama.com/library/llama4:scout) MoE model just to see what we find.

I picked this since they both came from Meta and likely had some similar lineages but also they were close enought in 
total trainable parameters that I could try some experimentation with some new normalization I've been thinking about.

Llama 4 scout has 16 experts with 17B active parameters per forward pass so it should be much faster and more energy 
efficient since the number of parameters per pass 17B versus 70b are quite differnet while the total trainable parameters are 
closer than some other test options we could come up with.

### Evaluation 
I ran our normal text prompt test suite across both models to generate the energy results. As expected at this scale the
Llama 4 scount model is much more energy efficient than the dense model.
| Model        | Total Time           | Total Watt Seconds | Total Tokens | Tokens per Watt Second | Tokens per Second | Watt Seconds per Response | Watt Seconds per Response per Trainable Parameter | Watt Seconds per Response per Executable Parameter |
|-------------|----------------------|--------------------|--------------|------------------------|-------------------|--------------------------|--------------------------------------------|--------------------------------------------|
| llama 3.3 70b | 0 days 00:12:09.38 | 216192.35       | 16597        | 0.07              | 22.75        | 5543.39              | 7.91E-08                               | 7.91E-08                               |
| llama 4 Scout | 0 days 00:03:56.69 | 64751.78       | 27092        | 0.41               | 114.45        | 1660.30              | 1.52E-08                               | 8.96E-19                               |

We can see here the significant energy savings across all the metrics with the tokens per watt-second value at 5.8x higher better energy efficiency 
for Scout (MoE) versus 3.3 (dense).  This is inspite of the llama 4 model having 1.5x more trainable parameters.  I generally perfer normalizing 
across token lenghths since that represents some of the core energy values regardless of verbosity of the models.  We see here that Scout is also 
quite a bit more verbose on the same prompt set than llama 3.3 which is a trend we've also seen with chain of thought models. We can also just normalize
across responses showing Scout 5.2x more energy efficient than 3.3.

Now lets explore what normalizing against different parameter counts looks like.  
