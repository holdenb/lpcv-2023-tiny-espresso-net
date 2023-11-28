# LPCV 2023 - Tiny Espresso Net

## Evaluation

### Format

- Input/output resolution: 512\*512
- Model Output: `14 * 512 * 512` for `Channel * Height * Width`. Each channel
  corresponds to the predicted probability for one category.

### Submission
- `solution.pyz`: the zipped package of solution/. [`zipapp`](https://docs.python.org/3/library/zipapp.html) should be used to compress the package.

Recommended command where solution is the name to your directory: `python3.6 -m zipapp  solution  -p='/usr/bin/env python3.6'`

### Metrics
- Accuracy: Dice Coefficient over all 14 categroies. As calculated in evaluation/Accuracy.py
- Speed: Average runtime for processing one frame (s/f). As calculated in solution/main.py
