# LLama-notes
### This repo should be transfer into a blog later
All notes from llama 3 training report
https://ai.meta.com/research/publications/the-llama-3-herd-of-models/
_______________

Model Architecture:

https://arxiv.org/pdf/2302.13971
1. **RMSNorm**: https://arxiv.org/pdf/1910.07467
   $$
   a = \sum_{j=1}^{m}w_{ij}x_{j} 
    $$
   $$
   \tilde{a_{i}} = \frac{a_{i}}{RMS(a)}g_{i}$$
   Where $$RMS(a) = \sqrt{\frac{1}{n}\sum_{i=1}^{m}a_{i}^{2}}$$
   **RMSNorm does not have re-centering, but it is not fundamental to the performance**
2. **SwiGLU**: https://arxiv.org/pdf/2002.05202
	$$(Swish_{1}(xW)\otimes xV)W_{2}
	$$
	$$
	Swish_{\beta}(x) = x\sigma({\beta}x)$$
	**SwishGLU has state of the art performance along the most benchmarkings**
3. **Rotary PE**: https://arxiv.org/pdf/2104.09864
