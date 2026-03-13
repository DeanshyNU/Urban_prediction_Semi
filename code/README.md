# Urban Temperature Prediction - Semi-Supervised Learning

Chicago urban temperature downscaling using 68 labeled + 200 unlabeled weather stations.

## Results

| Backbone | SSL Method | RMSE |
|----------|-----------|------|
| GNN | Mean Teacher | 0.0237 |
| GNN | Pi Model | 0.0235 |
| iTransformer | Mean Teacher | 0.0219 |
| TimeMixer | Mean Teacher | 0.0205 |
| ModernTCN | Mean Teacher | 0.0791 |

---

## Directory Structure

```
code/
├── downscale-gnn/              # Original GNN (supervised + semi-supervised)
│   ├── data.py                 # Supervised data loading (PyG graph datasets)
│   ├── network.py              # Supervised GNN model (GraphConv encoder-processor-decoder)
│   ├── run.py                  # Supervised training entry point
│   ├── predict.py              # Inference/prediction
│   ├── explainer.py            # Model explainability
│   ├── data_semi.py            # Semi-supervised data loading (labeled + unlabeled)
│   ├── network_semi.py         # Semi-supervised GNN (loss only on labeled nodes)
│   ├── run_semi.py             # Semi-supervised training entry point
│   └── utils.py                # Utilities
│
├── datalib/                    # Data processing package (consolidated)
│   ├── __init__.py
│   ├── preprocessing.py        # Wind merging, variable reordering, NaN handling, normalization, auxiliary variables
│   ├── generation.py           # PyG dataset creation (kNN-sparsified similarity graph, WRF/CLMS/Urban/Geo features)
│   ├── augmentation.py         # Augmentation: FeatureNoise/Scale/Shift/Clip/Zero, TimeNoise/Shift, TransformFixMatch
│   └── geo_features.py         # Geographic features (PCA or AvgPooling, unified normalization for labeled/unlabeled)
│
├── models/                     # Model package (consolidated)
│   ├── __init__.py
│   ├── gnn.py                  # GNN (SAGEConv encoder-processor-decoder)
│   └── backbones.py            # MLP, Transformer, CNN, LSTM, iTransformer, TimeMixer, ModernTCN, SMamba
│
├── trainers/                   # Training loop package (consolidated)
│   ├── __init__.py
│   ├── meanteacher.py          # Mean Teacher training loop
│   └── pimodel.py              # Pi Model training loop
│
├── run_gnn_meanteacher.py      # [consolidated] GNN + Mean Teacher entry point
├── run_gnn_pimodel.py          # [consolidated] GNN + Pi Model entry point
├── run_itransformer_meanteacher.py  # [consolidated] iTransformer + Mean Teacher entry point
├── run_timemixer_meanteacher.py     # [consolidated] TimeMixer + Mean Teacher entry point
├── run_moderntcn_meanteacher.py     # [consolidated] ModernTCN + Mean Teacher entry point
│
└── utils.py                    # RMSE, MinMax/MinMax_first_dim normalization, auxiliary variables, plotting
```

---

## How to Run

### Supervised GNN (baseline)
```bash
cd downscale-gnn
python run.py          # supervised, labeled only
python run_semi.py     # semi-supervised, labeled + unlabeled
```

### Multi-backbone Mean Teacher (consolidated)
```bash
python run_gnn_meanteacher.py           # GNN + Mean Teacher
python run_itransformer_meanteacher.py  # iTransformer + Mean Teacher
python run_timemixer_meanteacher.py     # TimeMixer + Mean Teacher
python run_moderntcn_meanteacher.py     # ModernTCN + Mean Teacher
```

### GNN Pi Model (consolidated)
```bash
python run_gnn_pimodel.py
```

---

## Consolidated Module Dependencies

```
run_{backbone}_{method}.py
  ├── datalib                  → preprocess_unlabeled_data, dataGen_ESTnet,
  │                              dataGen_unlabeled_ESTnet, TransformFixMatch,
  │                              genGeoFeatures, genGeoFeatures_unlabeled
  ├── models                   → GNN / iTransformerModel / TimeMixerModel / ModernTCNModel
  ├── trainers                 → train_meanteacher / train_pimodel, test functions, loadCheckPoint
  └── utils                    → RMSE, plotHist
```

## Data Requirements

Data files (not included) at `../data/`:
- WRF (54 dims), CLMS (3 dims), UrbanFeature (17 dims), GeoFeatures
- Total ~83 input features per station per timestep
- 6624 timesteps (2018 summer + 2019 summer)
