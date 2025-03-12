import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Criando um dataset de exemplo (usuário, item, nota)
data = [
    ("user1", "filmeA", 5),
    ("user2", "filmeB", 3),
    ("user1", "filmeC", 4),
    ("user3", "filmeA", 2),
    ("user2", "filmeC", 5),
    ("user3", "filmeB", 4),
]

# Definindo o formato do dataset
df = pd.DataFrame(data, columns=["user", "item", "rating"])
# Define um "leitor" (Reader) para interpretar os dados.
reader = Reader(rating_scale=(1, 5))
# Converte o DataFrame para um formato compatível com a biblioteca Surprise.
dataset = Dataset.load_from_df(df, reader)

# Dividindo os dados entre treino e teste
trainset, testset = train_test_split(dataset, test_size=0.2)

# Treinando o modelo SVD
model = SVD()
model.fit(trainset)

# Fazendo previsões
predictions = model.test(testset)

# Avaliando o modelo
print("RMSE:", accuracy.rmse(predictions))

# Fazendo uma recomendação personalizada
user_id = "user3"
item_id = "filmeC"
pred = model.predict(user_id, item_id)
print(f"Nota prevista para {user_id} no {item_id}: {pred.est:.2f}")
