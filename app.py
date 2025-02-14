from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Carrega os dados do modelo treinado com tratamento de exceção
try:
    with open('model.pkl', 'rb') as f:
        model_data = pickle.load(f)
except Exception as e:
    raise Exception("Erro ao carregar o arquivo model.pkl: " + str(e))

# Verifica se as chaves essenciais estão presentes
required_keys = ['train_df', 'itens_df', 'tfidf', 'tfidf_matrix', 'article_id_to_idx']
for key in required_keys:
    if key not in model_data:
        raise KeyError(f"A chave '{key}' não foi encontrada no model.pkl.")

train_df = model_data['train_df']
itens_df = model_data['itens_df']  # Deve estar no formato esperado
tfidf = model_data['tfidf']
tfidf_matrix = model_data['tfidf_matrix']
article_id_to_idx = model_data['article_id_to_idx']

def get_user_profile(user_id):
    """
    Calcula o vetor de perfil do usuário como a média dos vetores das notícias lidas.
    Retorna None se não houver histórico ou ocorrer erro.
    """
    try:
        user_histories = train_df[train_df['userId'] == user_id]['history_list']
    except Exception as e:
        app.logger.error(f"Erro ao acessar 'history_list' para userId {user_id}: {e}")
        return None

    article_ids = []
    for history in user_histories:
        # Garante que o history seja iterável
        try:
            article_ids.extend(history)
        except Exception as e:
            app.logger.error(f"Erro ao processar histórico para userId {user_id}: {e}")
            continue

    # Remove duplicatas
    article_ids = list(set(article_ids))
    if not article_ids:
        app.logger.info(f"Nenhum artigo encontrado para o userId {user_id}")
        return None

    # Obtém os índices dos artigos existentes no mapeamento
    indices = [article_id_to_idx[a] for a in article_ids if a in article_id_to_idx]
    if not indices:
        app.logger.info(f"Nenhum índice encontrado para os artigos do userId {user_id}")
        return None

    try:
        user_vector = tfidf_matrix[indices].mean(axis=0)
    except Exception as e:
        app.logger.error(f"Erro ao calcular a média dos vetores TF-IDF para userId {user_id}: {e}")
        return None

    # Converte para um array NumPy para evitar problemas com np.matrix
    return np.asarray(user_vector)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON inválido ou vazio"}), 400

    # Determina se a requisição usa 'userId' ou 'text'
    if 'userId' in data:
        user_id = data['userId']
        user_vector = get_user_profile(user_id)
        if user_vector is None:
            return jsonify({"error": "Nenhum histórico encontrado para este usuário"}), 404
        try:
            cosine_similarities = linear_kernel(user_vector, tfidf_matrix).flatten()
        except Exception as e:
            app.logger.error(f"Erro no cálculo de similaridades para userId {user_id}: {e}")
            return jsonify({"error": "Erro no cálculo de similaridades"}), 500
    elif 'text' in data:
        try:
            user_vector = tfidf.transform([data['text']])
            cosine_similarities = linear_kernel(user_vector, tfidf_matrix).flatten()
        except Exception as e:
            app.logger.error(f"Erro ao processar o texto: {e}")
            return jsonify({"error": "Erro ao processar o texto"}), 500
    else:
        return jsonify({"error": "Envie 'userId' ou 'text' na requisição."}), 400

    # Tenta selecionar os índices dos 6 artigos mais similares
    try:
        top_indices = cosine_similarities.argsort()[-6:][::-1]
    except Exception as e:
        app.logger.error(f"Erro ao ordenar similaridades: {e}")
        return jsonify({"error": "Erro ao ordenar similaridades"}), 500

    recommendations = []
    if 'userId' in data:
        try:
            # Cria um conjunto com os artigos já lidos pelo usuário
            user_history = set()
            for hist in train_df[train_df['userId'] == data['userId']]['history_list']:
                try:
                    user_history.update([str(a) for a in hist])
                except Exception as e:
                    app.logger.error(f"Erro ao converter histórico para string para userId {data['userId']}: {e}")
            # Seleciona artigos que não estão no histórico
            for idx in top_indices:
                try:
                    article_id = str(itens_df.iloc[idx]['Page'])
                    if article_id in user_history:
                        continue
                    recommendations.append({
                        'Page': article_id,
                        'Title': itens_df.iloc[idx]['Title']
                    })
                    if len(recommendations) >= 5:
                        break
                except Exception as e:
                    app.logger.error(f"Erro ao acessar os dados do artigo no índice {idx}: {e}")
                    continue
        except Exception as e:
            app.logger.error(f"Erro ao processar recomendações para userId {data['userId']}: {e}")
            return jsonify({"error": "Erro ao processar recomendações"}), 500
    else:
        for idx in top_indices[:5]:
            try:
                recommendations.append({
                    'Page': str(itens_df.iloc[idx]['Page']),
                    'Title': itens_df.iloc[idx]['Title']
                })
            except Exception as e:
                app.logger.error(f"Erro ao acessar os dados do artigo no índice {idx}: {e}")
                continue

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
