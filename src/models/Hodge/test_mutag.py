from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')

print(len(dataset))          # número de grafos
print(dataset.num_classes)   # número de classes
print(dataset[0])            # primeiro grafo — veja os atributos disponíveis
print(dataset[0].edge_attr)  # features das arestas

from torch_geometric.loader import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)
batch = next(iter(loader))   # pega o primeiro batch

from HodgeGNN import HodgeGNN
model = HodgeGNN(
    in_dim=4,
    hidden_dims=[32, 32],
    out_dim=2,
    normalize='symmetric',
    pooling='mean',
    dropout=0.5,
    residual=False
)

out = model(batch)
print(out.shape)   # deve ser [32, 2]
