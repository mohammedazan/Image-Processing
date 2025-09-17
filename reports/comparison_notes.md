# Comparative analysis report

Generated: 2025-09-17 13:23:23.430923

## Summary table

Top experiments by accuracy:



| experiment   |   accuracy |   balanced_accuracy |   macro_f1 |   prediction_time |
|:-------------|-----------:|--------------------:|-----------:|------------------:|
| mv_stack     |   0.477477 |            0.290119 |   0.262656 |               nan |
| bernoulli    | nan        |          nan        | nan        |               nan |
| pointnet     | nan        |          nan        | nan        |               nan |

---

## Highlights

- Best experiment: **mv_stack** — accuracy=0.4775, macro_f1=0.2627, time=nan

- Worst experiment: **pointnet** — accuracy=nan, macro_f1=nan

## Automated interpretation notes

- If a model predicts a single class for most samples, macro-F1 will be close to 0 while accuracy depends on class distribution.

- Balanced accuracy gives more insight when classes are imbalanced; prefer it when dataset is skewed.

- Check confusion matrices (plots) for which classes are confused and whether a model always predicts the same class.


## Per-experiment quick notes

### mv_stack

- accuracy=0.4775, macro_f1=0.2627

- Top precision classes: Roteiche(1.00), Douglasie(0.70), Buche(0.42)

- Top recall classes: Buche(0.81), Fichte(0.62), Douglasie(0.48)



### bernoulli

- accuracy=nan, macro_f1=nan



### pointnet

- accuracy=nan, macro_f1=nan


