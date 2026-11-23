# Dati

Il dataset **GTEA** non è incluso nel repository. È stato scaricato da OpenDataLab:
<https://opendatalab.com/OpenDataLab/GTEA/tree/main> (è necessario registrarsi e fare l'accesso).

Per alleggerire il download servono solo:
- i **video**;
- le **label** delle azioni (`gtea_labels_71`).

## Struttura attesa

```
data/
├── Videos/
│   ├── S1_Cheese_C1.mp4
│   ├── ...
│   └── S4_Tea_C1.mp4        # 28 video
└── gtea_labels_71/
    └── labels/
        ├── S1_Cheese_C1.txt
        ├── ...
        └── S4_Tea_C1.txt    # una label per video
```

I nomi dei video devono corrispondere a quelli dei file di label.