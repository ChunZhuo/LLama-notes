# LLama-notes
### This repo should be transfer into a blog later
All notes from llama 3 training report
https://ai.meta.com/research/publications/the-llama-3-herd-of-models/
_______________

Model Architecture:

https://arxiv.org/pdf/2302.13971
1. **RMSNorm**: https://arxiv.org/pdf/1910.07467
   $$a = \sum_{j=1}^{m}w_{ij}x_{j}$$
   
   $$\tilde{a_{i}} = \frac{a_{i}}{RMS(a)}g_{i}$$
   
   Where
   $$RMS(a) = \sqrt{\frac{1}{n}\sum_{i=1}^{m}a_{i}^{2}}$$
   
   **RMSNorm does not have re-centering, but it is not fundamental to the performance**
2. **SwiGLU**: https://arxiv.org/pdf/2002.05202
   
   $$(Swish_{1}(xW)\otimes xV)W_{2}$$

   $$Swish_{\beta}(x) = x\sigma({\beta}x)$$
   
   **SwishGLU has state of the art performance along the most benchmarkings** 

3. **Rotary PE**: https://arxiv.org/pdf/2104.09864
   Best approach before:

$$q_{m}^{T}k_{n} = x_{m}^{T}W_{q}^{T}W_{k}x_{n}+x_{m}^{T}W_{q}^{T}W_{k} \tilde{p}{m-n}+\tilde{p}{m-n}^{T}W_{q}^{T}W_{k}x_{n}$$

   RoPE: Derive relative positional encoding from attention.

$$R_{\Theta,m}^{d}\mathbf{x}=
   \begin{pmatrix} x_1 \\ 
   x_2 \\ 
   x_3 \\ 
   x_4 \\ 
   \vdots \\ 
   x_{d-1} \\ 
   x_d 
   \end{pmatrix} 
   \otimes 
   \begin{pmatrix} \cos m\theta_1 \\ 
   \cos m\theta_1 \\ 
   \cos m\theta_2 \\ 
   \cos m\theta_2 \\ 
   \vdots \\ 
   \cos m\theta_{d/2} \\ 
   \cos m\theta_{d/2} 
   \end{pmatrix} 
   +
\begin{pmatrix} -x_2 \\ 
   x_1 \\ 
   -x_4 \\ 
   x_3 \\ 
   \vdots \\ 
   -x_d \\ 
   x_{d-1} 
   \end{pmatrix} 
   \otimes 
   \begin{pmatrix} 
   \sin m\theta_1 \\ 
   \sin m\theta_1 \\ 
   \sin m\theta_2 \\ 
   \sin m\theta_2 \\ 
   \vdots \\ 
   \sin m\theta_{d/2} \\ 
   \sin m\theta_{d/2} 
   \end{pmatrix}$$

