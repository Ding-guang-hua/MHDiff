- TikTok

```python
python Main.py --data tiktok --reg 1e-4 --trans 1 --cl_method 1 --steps 50 --temp 0.1
```

- Baby

```python
python Main.py --data baby --ssl_reg 1e-1 --keepRate 1 --epoch 100 --gnn_layer 2
```

- Sports

```python
python Main.py --data sports --reg 1e-6 --temp 0.3 --ris_lambda 0.1 --e_loss 0.01 --keepRate 1 --trans 1 --epoch 130 --cl_method 1 --rebuild_k 4
```

