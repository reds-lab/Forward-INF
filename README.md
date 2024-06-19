# The Mirrored Influence Hypothesis

This repository contains the source code for the paper titled "The Mirrored Influence Hypothesis: Efficient Data Influence Estimation by Harnessing Forward Passes", 
published at CVPR 2024.

 | [arXiv](https://arxiv.org/pdf/2402.08922) | 

## Verification of the Hypothesis

This section outlines the steps to verify the Mirrored Influence Hypothesis. 

### Convex Models
1. **Execution of Scripts:**
   - Begin by running the script `python LOO-DualLOO-Convex.py`. 
2. **Analysis:**
   - After running the script, proceed with the analysis using the Jupyter Notebook:
     - `LOO-DualLOO-Convex_Analysis.ipynb`

### Non-Convex Models
1. **Analysis:**
   - Use the Jupyter Notebook provided for non-convex models:
     - `LOO-DualLOO-Group-Nonconvex-mnist.ipynb`

## Applications


## Citation

If you find "The Mirrored Influence Hypothesis" useful in your research, please consider citing:

```bibtex
@article{ko2024mirrored,
  title={The Mirrored Influence Hypothesis: Efficient Data Influence Estimation by Harnessing Forward Passes},
  author={Ko, Myeongseob and Kang, Feiyang and Shi, Weiyan and Jin, Ming and Yu, Zhou and Jia, Ruoxi},
  journal={arXiv preprint arXiv:2402.08922},
  year={2024}
}
