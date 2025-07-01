# Lightweight Encoder–Transformer with 1D Convolutional Projections

This implementation injects **sketched learnable kernel functions** into a Multi‑Head Attention layer, replacing softmax, and applies it to the **CMAPSS NASA dataset**. It demonstrates both accuracy and latency improvements on the **FD002** and **FD004** subsets.

<div align="center">
  <img src="https://github.com/user-attachments/assets/f4b40910-9bd6-48e5-941b-285bec11772d" alt="SketchAttn" width="60%"/>
  <p><em>Figure 1: Multi‑Head Attention with 1D convolutional projections and learnable kernels.</em></p>
</div>

---

## Dataset

I use the **Turbofan Engine Degradation Simulation (C‑MAPSS)** dataset [Saxena et al., 2008] with specific focus on FD002 and FD004 to highlight improvements in prediction accuracy and inference latency.

---

## Results

### Table 1: Softmax vs. Sketched Kernel Accuracy (lower is better)

| Method           | FD001 RMSE / Score | FD002 RMSE / Score | FD003 RMSE / Score | FD004 RMSE / Score |
|------------------|--------------------|--------------------|--------------------|--------------------|
| **Softmax**      | 13.34 / 280.7      | 14.38 / 935.57     | 12.94 / 294.65     | 17.83 / 1931.56    |
| **Sketched Kernel** | 13.59 / 258.89     | 13.19 / 685.31     | 13.17 / 332.02     | 16.12 / 1417.73    |

> *Note:* As dataset size and sequence length grow, the sketched kernel’s $\(O(n \cdot r)\)$ complexity $(with \(r \ll n\))$ yields both accuracy and speed advantages over softmax.

---

### Table 2: Function Execution Speed (lower is better)

| Function         | Subset | Calculation Speed (ms) |
|------------------|--------|------------------------:|
| **Kernel-based** | FD001  |                  117.82 |
|                  | FD002  |                  327.42 |
|                  | FD003  |                  131.96 |
|                  | FD004  |                  325.40 |
| **Softmax**      | FD001  |                  119.45 |
|                  | FD002  |                  394.44 |
|                  | FD003  |                  119.45 |
|                  | FD004  |                  401.56 |

> *Experiments were run single‑threaded on an Intel Core i5‑12500.*

---

# References

1. **Attention Is All You Need**  
   Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomes, A., Kaiser, Ł., & Polosukhin, I. (2017). In _Advances in Neural Information Processing Systems_, 30, 5998–6008.

2. **Linear Transformers with Learnable Functions Are Better In-Context Models**  
   Aksenov, Y., Balagansky, N., Lo Cicero Vaina, S., Shaposhnikov, B., Gorbatovski, A., & Gavrilov, D. (2024). In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics_ (Long Papers, pp. 9584–9597).

3. **Turbofan Engine Degradation Simulation Data Set (C‑MAPSS)**  
   Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). In _2008 PHM Society Annual Conference_, pp. 1–9.

4. **PolySketchFormer: Fast Transformers via Sketching Polynomial Kernels**  
   Kacham, P., Mirrokni, V., & Zhong, P. (2024). In _Proceedings of the 41st International Conference on Machine Learning_.

5. **Sketching as a Tool for Numerical Linear Algebra**  
   Woodruff, D. P. (2014). _Foundations and Trends® in Theoretical Computer Science_, 10(1–2), 1–157.
