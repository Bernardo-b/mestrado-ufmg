# %%
import numpy as np
import re
import matplotlib.pyplot as plt

def plot_rota_particula(coords, vetor_posicao, titulo='Rota da Partícula'):
    rota = np.argsort(vetor_posicao)
    caminho = coords[rota]
    caminho = np.vstack([caminho, caminho[0]])
    plt.figure(figsize=(8, 6))
    plt.plot(caminho[:, 0], caminho[:, 1], '-o', markersize=5)
    plt.title(titulo)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis('equal')
    plt.show()


# %% [markdown]
# Importação dos dados

# %%

with open("data/att48.tsp", "r") as f:
    lines = f.readlines()

coords = []
start = False
for line in lines:
    if "NODE_COORD_SECTION" in line:
        start = True
        continue
    if start:
        if "EOF" in line or line.strip() == "":
            break
        parts = re.findall(r"[\d\.\-]+", line)
        if len(parts) >= 3:
            coords.append((float(parts[1]), float(parts[2])))

coords = np.array(coords)


# %% [markdown]
# Cálculo da matriz de distâncias entre os pontos

# %%
n_cidades = len(coords)
dist_matrix = np.zeros((n_cidades, n_cidades))

for i in range(n_cidades):
    for j in range(n_cidades):
        if i != j:
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])


# %% [markdown]
## PSO

# %%
n_particulas = 30
enxame = []

for _ in range(n_particulas):
    posicao = np.random.rand(n_cidades)
    velocidade = np.random.uniform(-0.1, 0.1, size=n_cidades)
    rota = np.argsort(posicao)
    custo = sum(dist_matrix[rota[i], rota[(i + 1) % n_cidades]] for i in range(n_cidades))

    particula = {
        'posicao': posicao,
        'velocidade': velocidade,
        'pbest': posicao.copy(),
        'pbest_custo': custo
    }

    enxame.append(particula)

gbest = min(enxame, key=lambda p: p['pbest_custo'])
gbest_pos = gbest['pbest'].copy()
gbest_custo = gbest['pbest_custo']

n_iter = 10000


c1 = 1.8
c2 = 1.5

w_max = 0.9
w_min = 0.6

historico_gbest = []
iter_atual = 0

for iter_atual_loop in range(n_iter):
    w = w_max - (w_max - w_min) * (iter_atual_loop / n_iter)

    for p in enxame:
        r1_vec = np.random.rand(n_cidades)
        r2_vec = np.random.rand(n_cidades)

        termo_inercia = w * p['velocidade']
        termo_cognitivo = c1 * r1_vec * (p['pbest'] - p['posicao'])
        termo_social = c2 * r2_vec * (gbest_pos - p['posicao'])

        p['velocidade'] = termo_inercia + termo_cognitivo + termo_social

        p['posicao'] += p['velocidade']
        
        p['posicao'] = np.clip(p['posicao'], 0.0, 1.0)

        rota = np.argsort(p['posicao'])
        custo = sum(dist_matrix[rota[i], rota[(i + 1) % n_cidades]] for i in range(n_cidades))

        if custo < p['pbest_custo']:
            p['pbest'] = p['posicao'].copy()
            p['pbest_custo'] = custo

    melhor_particula_iter = min(enxame, key=lambda p: p['pbest_custo'])
    if melhor_particula_iter['pbest_custo'] < gbest_custo:
        gbest_custo = melhor_particula_iter['pbest_custo']
        gbest_pos = melhor_particula_iter['pbest'].copy()

    if historico_gbest:
        historico_gbest.append(min(gbest_custo, historico_gbest[-1]))
    else:
        historico_gbest.append(gbest_custo)
# ... código existente ...

# Plotar a convergência
plt.figure(figsize=(10, 6))
plt.plot(historico_gbest)
plt.title('Convergência do PSO - Melhor Custo por Iteração')
plt.xlabel('Iteração')
plt.ylabel('Melhor Custo (gbest)')
plt.grid(True)
plt.show()

# Opcional: Imprimir o melhor custo e a melhor rota encontrada
print(f"\nMelhor custo encontrado: {gbest_custo}")
melhor_rota_final = np.argsort(gbest_pos)
print(f"Melhor rota encontrada: {melhor_rota_final}")
plot_rota_particula(coords, gbest_pos, titulo=f"Melhor Rota Encontrada (Custo: {gbest_custo:.2f})")


### DE

# %%
n_populacao = 30
n_iter = 10000
F = 0.5
CR = 0.9
lambda_param = 0.1

populacao = []
for _ in range(n_populacao):
    individuo_pos = np.random.rand(n_cidades)
    rota = np.argsort(individuo_pos)
    custo = sum(dist_matrix[rota[i], rota[(i + 1) % n_cidades]] for i in range(n_cidades))
    populacao.append({'posicao': individuo_pos, 'custo': custo})

melhor_global = min(populacao, key=lambda ind: ind['custo'])
gbest_pos = melhor_global['posicao'].copy()
gbest_custo = melhor_global['custo']

historico_gbest = [gbest_custo]

for geracao in range(n_iter):
    nova_populacao = []
    for i in range(n_populacao):
        alvo = populacao[i]

        idxs = [idx for idx in range(n_populacao) if idx != i]
        np.random.shuffle(idxs)
        if len(idxs) < 2: # Precisamos apenas de r2 e r3 distintos de i e entre si
            vetor_mutante = alvo['posicao'].copy()
        else:
            r2_idx, r3_idx = idxs[0], idxs[1] # idxs já não contém i
            b_de, c_de = populacao[r2_idx], populacao[r3_idx] # Renomeado para não conflitar
            vetor_mutante = alvo['posicao'] + lambda_param * (gbest_pos - alvo['posicao']) + F * (b_de['posicao'] - c_de['posicao'])
            vetor_mutante = np.clip(vetor_mutante, 0.0, 1.0)

        # if len(idxs) < 3:
        #     vetor_mutante = alvo['posicao'].copy()
        # else:
        #     a, b, c = populacao[idxs[0]], populacao[idxs[1]], populacao[idxs[2]]
        #     vetor_mutante = a['posicao'] + F * (b['posicao'] - c['posicao'])
        #     vetor_mutante = np.clip(vetor_mutante, 0.0, 1.0)

        vetor_ensaio = np.zeros_like(alvo['posicao'])
        j_rand = np.random.randint(0, n_cidades)

        for j in range(n_cidades):
            if np.random.rand() < CR or j == j_rand:
                vetor_ensaio[j] = vetor_mutante[j]
            else:
                vetor_ensaio[j] = alvo['posicao'][j]
        
        rota_ensaio = np.argsort(vetor_ensaio)
        custo_ensaio = sum(dist_matrix[rota_ensaio[k], rota_ensaio[(k + 1) % n_cidades]] for k in range(n_cidades))

        if custo_ensaio < alvo['custo']:
            nova_populacao.append({'posicao': vetor_ensaio, 'custo': custo_ensaio})
            if custo_ensaio < gbest_custo:
                gbest_custo = custo_ensaio
                gbest_pos = vetor_ensaio.copy()
        else:
            nova_populacao.append(alvo)

    populacao = nova_populacao

    historico_gbest.append(gbest_custo)


# Resultados finais
print(f"\nMelhor custo encontrado pelo DE: {gbest_custo}")
melhor_rota_final_de = np.argsort(gbest_pos)
print(f"Melhor rota encontrada pelo DE: {melhor_rota_final_de}")

# Plotar convergência (semelhante ao que você já tem para PSO)
plt.figure(figsize=(10, 6))
plt.plot(historico_gbest)
plt.title('Convergência do DE - Melhor Custo por Geração')
plt.xlabel('Geração')
plt.ylabel('Melhor Custo (gbest)')
plt.grid(True)
plt.show()

plot_rota_particula(coords, gbest_pos, titulo=f"Melhor Rota DE (Custo: {gbest_custo:.2f})")

## AG
# %%
import numpy as np
import random


def calcular_custo_rota(rota, dist_matrix):
    custo = 0
    for i in range(len(rota)):
        cidade_atual = rota[i]
        proxima_cidade = rota[(i + 1) % len(rota)]
        custo += dist_matrix[cidade_atual, proxima_cidade]
    return custo

def calcular_limites_custo(dist_matrix, n_cidades):
    custo_min_possivel = 0
    custo_max_possivel = 0
    for i in range(n_cidades):
        col_sem_zeros = dist_matrix[i, dist_matrix[i, :] > 0]
        if len(col_sem_zeros) > 0:
            custo_min_possivel += np.min(col_sem_zeros) if len(col_sem_zeros) > 0 else 0
        custo_max_possivel += np.max(dist_matrix[i, :]) if np.max(dist_matrix[i, :]) > 0 else 0
    return custo_min_possivel, custo_max_possivel

# --- Operadores Genéticos ---
def selecao_torneio(populacao_com_fitness, k=2):
    selecionados = []
    for _ in range(len(populacao_com_fitness)):
        participantes = random.sample(populacao_com_fitness, k)
        vencedor = min(participantes, key=lambda x: x['fitness_escalonada']) # Minimizando fitness escalonada
        selecionados.append(vencedor['individuo'])
    return selecionados

def crossover_pmx(pai1, pai2):
    n = len(pai1)
    filho1, filho2 = [-1]*n, [-1]*n
    
    ponto1, ponto2 = sorted(random.sample(range(n), 2))

    mapeamento1, mapeamento2 = {}, {}
    
    for i in range(ponto1, ponto2 + 1):
        filho1[i] = pai1[i]
        filho2[i] = pai2[i]
        mapeamento1[pai1[i]] = pai2[i]
        mapeamento2[pai2[i]] = pai1[i]

    for i in list(range(ponto1)) + list(range(ponto2 + 1, n)):
        val_pai2 = pai2[i]
        while val_pai2 in mapeamento1:
            val_pai2 = mapeamento1[val_pai2]
        filho1[i] = val_pai2

        val_pai1 = pai1[i]
        while val_pai1 in mapeamento2:
            val_pai1 = mapeamento2[val_pai1]
        filho2[i] = val_pai1
        
    return list(filho1), list(filho2)

def mutacao_swap(individuo):
    idx1, idx2 = random.sample(range(len(individuo)), 2)
    individuo[idx1], individuo[idx2] = individuo[idx2], individuo[idx1]
    return individuo

def mutacao_insert(individuo):
    idx1, idx2 = sorted(random.sample(range(len(individuo)), 2))
    cidade_movida = individuo.pop(idx2)
    individuo.insert(idx1, cidade_movida)
    return individuo

def mutacao_inversion(individuo):
    idx1, idx2 = sorted(random.sample(range(len(individuo)), 2))
    if idx1 == idx2: return individuo # Evita inversão de um único elemento
    segmento = individuo[idx1:idx2+1]
    segmento.reverse()
    individuo[idx1:idx2+1] = segmento
    return individuo


# %%
n_populacao_ag = 70
n_geracoes_ag = 5000
taxa_crossover_ag = 0.8
taxa_mutacao_ag = 0.08
k_torneio = 2

custo_min_teorico, custo_max_teorico = calcular_limites_custo(dist_matrix, n_cidades)
if custo_min_teorico == custo_max_teorico:
    custo_max_teorico = custo_min_teorico + 1 

populacao_ag = []
for _ in range(n_populacao_ag):
    individuo = list(np.random.permutation(n_cidades))
    populacao_ag.append(individuo)

melhor_individuo_global_ag = None
melhor_custo_global_ag = float('inf')
historico_melhor_custo_ag = []

for geracao in range(n_geracoes_ag):
    populacao_com_custos = []
    for ind in populacao_ag:
        custo = calcular_custo_rota(ind, dist_matrix)
        populacao_com_custos.append({'individuo': ind, 'custo_real': custo})

    populacao_com_fitness_escalonada = []
    for item in populacao_com_custos:
        fitness_esc = (item['custo_real'] - custo_min_teorico) / (custo_max_teorico - custo_min_teorico)
        fitness_esc = np.clip(fitness_esc, 0.0, 1.0) 
        populacao_com_fitness_escalonada.append({
            'individuo': item['individuo'],
            'custo_real': item['custo_real'],
            'fitness_escalonada': fitness_esc
        })
        
        if item['custo_real'] < melhor_custo_global_ag:
            melhor_custo_global_ag = item['custo_real']
            melhor_individuo_global_ag = item['individuo'][:]

    historico_melhor_custo_ag.append(melhor_custo_global_ag)

    pais_selecionados = selecao_torneio(populacao_com_fitness_escalonada, k_torneio)
    
    proxima_geracao = []

    if melhor_individuo_global_ag:
        melhor_da_geracao_atual = min(populacao_com_fitness_escalonada, key=lambda x: x['custo_real'])
        proxima_geracao.append(melhor_da_geracao_atual['individuo'][:])

    i = 0
    while len(proxima_geracao) < n_populacao_ag:
        pai1 = pais_selecionados[i % len(pais_selecionados)]
        pai2 = pais_selecionados[(i + 1) % len(pais_selecionados)]
        i += 2

        filho1_cand, filho2_cand = list(pai1), list(pai2)

        if random.random() < taxa_crossover_ag:
            filho1_cand, filho2_cand = crossover_pmx(list(pai1), list(pai2))

        if random.random() < taxa_mutacao_ag:
            tipo_mutacao = random.choice(['swap', 'insert', 'inversion'])
            if tipo_mutacao == 'swap':
                filho1_cand = mutacao_swap(filho1_cand)
            elif tipo_mutacao == 'insert':
                filho1_cand = mutacao_insert(filho1_cand)
            else:
                filho1_cand = mutacao_inversion(filho1_cand)
        
        if random.random() < taxa_mutacao_ag:
            tipo_mutacao = random.choice(['swap', 'insert', 'inversion'])
            if tipo_mutacao == 'swap':
                filho2_cand = mutacao_swap(filho2_cand)
            elif tipo_mutacao == 'insert':
                filho2_cand = mutacao_insert(filho2_cand)
            else:
                filho2_cand = mutacao_inversion(filho2_cand)
        
        if len(proxima_geracao) < n_populacao_ag:
            proxima_geracao.append(filho1_cand)
        if len(proxima_geracao) < n_populacao_ag:
            proxima_geracao.append(filho2_cand)
            
    populacao_ag = proxima_geracao


print(f"\nMelhor custo AG: {melhor_custo_global_ag}")
print(f"Melhor rota AG: {melhor_individuo_global_ag}")

# Plotar convergência
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(historico_melhor_custo_ag)
plt.title('Convergência do AG - Melhor Custo por Geração')
plt.xlabel('Geração')
plt.ylabel('Melhor Custo')
plt.grid(True)
plt.show()

# Se tiver a função plot_rota_particula e coords:
def plot_rota_individuo(coords_plot, individuo_rota, titulo='Rota AG'):
    caminho = coords_plot[individuo_rota]
    caminho = np.vstack([caminho, caminho[0]]) # Volta para o início
    plt.figure(figsize=(8, 6))
    plt.plot(caminho[:, 0], caminho[:, 1], '-o', markersize=5)
    plt.title(titulo)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis('equal')
    plt.show()
plot_rota_individuo(coords, melhor_individuo_global_ag, titulo=f"Melhor Rota AG (Custo: {melhor_custo_global_ag:.2f})")


# %%



