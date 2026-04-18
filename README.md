# AMBHFN

Adaptive Multi-Branch Heterogeneous Fusion Wind Prediction Network

## Project Introduction

This project is a deep learning model for wind field prediction in Northeast China, using the AMBHFN architecture. The model combines 3D U-Net, iTransformer-based multi-agent system with graph convolution, Multi-Feature Identity-Enhanced Wind Prediction Network and Adaptive Fusion mechanisms to achieve accurate prediction of wind field spatiotemporal sequences.

## Environment Requirements

- Python 3.8+
- PyTorch 1.9.0+
- CUDA (recommended for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

- torch>=1.9.0
- numpy
- xarray
- pandas
- matplotlib
- scikit-learn
- einops
- thop
- tqdm
- PyWavelets

## Data Preparation

### Data Format

- Input data: UV wind components (u, v) and auxiliary meteorological data (z, t)
- Terrain data: DEM digital elevation model
- Data shape: (time, channels, height, width)
- Default resolution: 64×80

### Data Files

- `uv100_train.npy` / `uv100_test.npy`: Training/testing wind field data
- `1000zt_train.npy` / `1000zt_test.npy`: Training/testing height field data
- `DEM_northeast.npy`: Northeast region terrain data

## Usage

### 1. Train Model

```bash
python main.py
```

The trained model will be saved to `chkfile/checkpoint_ambhfn.chk`.

Training configurations can be modified in `config.py`:

- `batch_size`: Batch size
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate
- `patience`: Early stopping patience

### 2. Evaluate Model

```bash
python Table_RMSE_MAE_ACC_WDFA.py
```

This will output evaluation metrics including:
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **ACC**: Anomaly Correlation Coefficient
- **WDFA**: Wind Direction Forecast Accuracy at different thresholds (90°, 45°, 22.5°)

## Model Checkpoint

During training, the model is automatically saved to `chkfile/checkpoint_ambhfn.chk`.

## Evaluation Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **ACC**: Anomaly Correlation Coefficient
- **WDFA**: Wind Direction Forecast Accuracy

## Notes

1. Ensure data file paths are correctly configured
2. Using GPU training can significantly speed up the process
3. Adjust `batch_size` based on GPU memory
4. Check device configuration in `config.py` before first run

## License

[MIT LICENSE](LICENSE)
