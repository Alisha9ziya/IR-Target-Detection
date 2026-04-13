# IR Small Target Detection

## Run

python src/main.py

## Pipeline

* Grayscale conversion
* Top-hat filtering
* Density + peak detection
* Region growing
* Mask combination

## Metrics

* PD (Detection Rate)
* FA (False Alarm Rate)

## Results

PD ≈ 0.6–0.9
FA ≈ 0.001–0.01

## Output

Saved in results/outputs/
