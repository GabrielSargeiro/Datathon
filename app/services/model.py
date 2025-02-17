# app/services/model.py
import pickle
from sklearn.metrics.pairwise import linear_kernel


def load_model(path='model.pkl'):
    try:
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
    except Exception as e:
        raise Exception("Erro ao carregar model.pkl: " + str(e))

    # Verifica se as chaves essenciais estão presentes
    required_keys = ['train_df', 'itens_df', 'tfidf', 'tfidf_matrix', 'article_id_to_idx']
    for key in required_keys:
        if key not in model_data:
            raise KeyError(f"A chave '{key}' não foi encontrada no model.pkl.")
    return model_data


def recommend_from_model(model_data, user_input, top_n=5):
    """
    Gera recomendações usando o modelo carregado.
    O parâmetro `user_input` pode ser, por exemplo, uma string criada a partir do histórico do usuário.
    """
    tfidf = model_data['tfidf']

    try:
        # Transforma o input usando o TF-IDF do modelo
        input_vector = tfidf.transform([user_input])
    except Exception as e:
        raise Exception("Erro ao transformar o input com TF-IDF: " + str(e))

    try:
        # Calcula similaridades entre o input e a matriz TF-IDF dos itens
        cosine_similarities = linear_kernel(input_vector, model_data['tfidf_matrix']).flatten()
        # Seleciona os índices dos itens mais similares
        top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    except Exception as e:
        raise Exception("Erro no cálculo das similaridades: " + str(e))

    recommendations = []
    for idx in top_indices:
        try:
            article = model_data['itens_df'].iloc[idx]
            recommendations.append({
                'Page': str(article['Page']),
                'Title': article['Title']
            })
        except Exception as e:
            continue
    return recommendations
