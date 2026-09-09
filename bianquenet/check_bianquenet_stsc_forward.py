import torch

from project_paths import PROJECT_ROOT
from models.bianquenet import BianqueNetMiniSTSC


model = BianqueNetMiniSTSC()
model.eval()

x = torch.randn(1, 1, 384, 384)

with torch.no_grad():
    y = model(x)

print("input shape :", tuple(x.shape))
print("output shape:", tuple(y.shape))

assert y.shape == (1, 4, 384, 384)
print("BianqueNetMiniSTSC forward test passed")
