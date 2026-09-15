list1 = ['LENEP', 'ZNF415', 'KLHDC8A', 'OSGIN1', 'TSPAN32', 'CDKN2C', 'FASTK', 'NEUROD2', 'BNIP3L', 'PEX5', 'HOXD1', 'LAGE3', 'SCN11A', 'TIPRL', 'DNAJB14']
list2 = ['KCNB1', 'KRT18', 'VIPR1', 'SMYD5', 'SPINK1', 'CETN3', 'CYP2A13', 'DNAJB14', 'RIMBP2', 'MAPKAPK2', 'ARMC1', 'SERHL2', 'GABRB3', 'ZNF415', 'HGF']
list3 = ["DNAJB14", "TOPORS", "FGF23", "BAP1", "TCHH", "MBD4", "RIMBP2", "DIP2A", "HGF", "WT1", "IL1RL1", "TESK1", "RECQL", "GAS8", "PRR11"]

# Finding the intersection (common genes) across all three lists
common_genes = list(set(list2) & set(list3))

print("Common genes:", common_genes)