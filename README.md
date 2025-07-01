A lightweight implementation of the encoder-transformer with 1-D convolutional projections of querries, keys and values with sketched learnable kernel functions on the CMAPSS NASA dataset. 
I am currently preparing an article about this method. Better to use with FD002 and FD004 as it shows both accuracy and latency improvement on these subsets in comparison with the softmax function.

![image](![SketchAttn](https://github.com/user-attachments/assets/f4b40910-9bd6-48e5-941b-285bec11772d)

fig.1: A Multi-Head Attention layer.

References:

Vaswani A., Shazeer N., Parmar N., Uszkoreit J., Jones L., Gomes A., Kaiser L., Polosukhin I. Attention Is All You Need. In: Advances in 
Neural Information Processing Systems (NeurIPS), 2017, pp. 5998–6088.

Aksenov Y., Balagansky N., Lo Cicero Vaina S., Shaposhnikov B., Gorbatovski A., Gavrilov D. Linear Transformers with Learnable Functions are Better In-Context Models. 
In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL, Long Papers), 2024, pp. 9584–9597.

Saxena A., Goebel K., Simon D., Eklund N. Turbofan Engine Degradation Simulation Data Set (C‑MAPSS). In: 2008 PHM Society Annual Conference, 2008, pp. 1–9.

Kacham P., Mirrokni V., Zhong P. PolySketchFormer: Fast Transformers via Sketching Polynomial Kernels. In: Proceedings of the 41st International Conference on Machine Learning (ICML), 2024.

Woodruff D. P. Sketching as a Tool for Numerical Linear Algebra. Foundations and Trends® in Theoretical Computer Science, vol. 10, no. 1–2, 2014, pp. 1–157.

