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

I picked this since they both came from Meta and likely had some similar lineages but also they were close enough in 
total trainable parameters that I could try some experimentation with some new normalization I've been thinking about.

Llama 4 scout has 16 experts with 17B active parameters per forward pass so it should be much faster and more energy 
efficient since the number of parameters per pass 17B versus 70b are quite differnet while the total trainable parameters are 
closer than some other test options we could come up with. Also we are well above the threashold discussed in part 1 where we should
easily be overcoming the overhead to select and average experts.

### Evaluation 
I ran our normal text prompt test suite across both models to generate the energy results. As expected, the
Llama 4 scount model is much more energy efficient than the dense model.

| Model        | Total Time           | Total Watt Seconds | Total Tokens | Tokens per Watt Second | Tokens per Second | Watt Seconds per Response | 
|-------------|----------------------|--------------------|--------------|------------------------|-------------------|--|
| llama 3.3 70b | 0 days 00:12:09.38 | 216192.35       | 16597        | 0.07              | 22.75        | 5543.39              |
| llama 4 Scout | 0 days 00:03:56.69 | 64751.78       | 27092        | 0.41               | 114.45        | 1660.30              |

We can see here the significant energy savings across all the metrics with the tokens per watt-second value at 5.8x higher better energy efficiency 
for Scout (MoE) versus 3.3 (dense).  This is inspite of the llama 4 model having 1.5x more trainable parameters.  I generally perfer normalizing 
across token lenghths since that represents some of the core energy values regardless of verbosity of the models.  We also see that Scout is 
quite a bit more verbose on the same prompt set than Llama 3.3 which is a trend we've also seen with chain of thought models. We can also just normalize
across responses showing Scout 3.3x more energy efficient than Llama 3.3 (still alot better given its longer responses) per average response.

## What can we learn if we normalize against different parameter properties?

I have been curious as to whether just knowing the number of active parameters of new models as they are released could give us much insight in to 
their energy efficiency.  This is a good opportunity to evaluate that.

| Model        | Watt Seconds per Response per Trainable Parameter | Watt Seconds per Response per Active Parameter |
|---------------------------------------------------------|--------------------------------------------|----|
| llama 3.3 70b | 7.91E-08                               | 7.91E-08                               |
| llama 4 Scout | 1.52E-08                               | 9.76E-08                              |

After normlizaing against trainable parameters we see that Scout is again 5.2x more energy efficient than Llama 3.3 
and normalizing against active parameters its about 80% as energy efficient as Llama 3.3.  So this is the first normalization where we are seeing the relative 
energy show up more for the dense model.  We have also got confounding results when normalizing against different values.  

Given we learned about the MoE overhead in part one I have a hypothesis that if we control for tokens output
and executable parameters then MoE should have less energy efficiency than dense models based on what we learned in part 1.  

Lets see:

| Model        | Watt Seconds per Token per Active Parameter | 
|---------------------------------------------------------|--------------------------------------------|
| llama 3.3 70b | 1.86E-10                               | 
| llama 4 Scout | 1.41E-10                               | 

Humm, well I guess my hypothesis didn't hold.  When normalized against token output and active parameters Scout is about 1.3x more energy efficient than
Llama 3.3.  I don't currently have any good ideas on why that might be except that there are likely additional optimizations in Llama 4 which are 
improving the token output performance and efficiency beyond the overhead of the MoE selection.

I also beleive due to the confounding results when normalizing and the lack of key takaways from this analysis I'm not very confident that we can easily make
estimations of future models energy efficiencies from their high level parameter values (trainable/active).

## Takeaways
In this part we did demonstrate that at these production level SotA scales (70b+) that the overhead of MoE expert selection is well below the performance gains
from the fewer calculations.  We also demonstrated that attempting to compare models by normalizing against active or trainable parameters probably isn't
adding alot of insight in to efficiency calculations or enabling improved business decisions around efficiency.  We'll have to keep measuring models instance
by instance as they are released to get energy efficiency insights.

## Notes
Tests were conducted on fixed sets of prompts against Nvidia A100 80gb using Ollama models all qantized and served in the same format.  Test were hosted on 
[Crusoe Cloud](https://www.crusoe.ai/).  All energy values are GPU only. While internally we've migrated to doing most of our tests using vllm I chose to use Ollama for this due to some challenges getting comparable quantized models installed and running easily.


