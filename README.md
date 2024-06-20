# The Mirrored Influence Hypothesis

This repository contains the source code for the paper titled "The Mirrored Influence Hypothesis: Efficient Data Influence Estimation by Harnessing Forward Passes", 
published at CVPR 2024.

 | [arXiv](https://arxiv.org/pdf/2402.08922) | 


## Environment Setup

1. **Create and Activate the Conda Environment:**
     ```bash
     conda create -n data-infl python=3.8.16
     ```
     ```bash
     conda activate data-infl
     ```
     ```bash
     pip install -r requirements.txt
     ```

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

This section provides an example of applying our algorithm in one of our applications (e.g., data leakage experiment). 
- To review the implementation, refer to the provided Jupyter Notebook in the data-leakage directory:
  - `FINF-Duplication-ResNet18-main.ipynb`
- The same codebase can be adapted for various applications.

## Contact Information

For any inquiries, issues, or contributions, please contact:

- **Myeongseob Ko**: `myeongseob@vt.edu`

Feel free to reach out if you have any questions.

## Citation

If you find "The Mirrored Influence Hypothesis" useful in your research, please consider citing:

```bibtex
@article{ko2024mirrored,
  title={The Mirrored Influence Hypothesis: Efficient Data Influence Estimation by Harnessing Forward Passes},
  author={Ko, Myeongseob and Kang, Feiyang and Shi, Weiyan and Jin, Ming and Yu, Zhou and Jia, Ruoxi},
  journal={arXiv preprint arXiv:2402.08922},
  year={2024}
}
