from fastapi import APIRouter, HTTPException
from app.models.data_models import RecommendationInput, RecommendationOutput
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack

router = APIRouter(prefix="/recommendation", tags=["Recommendation"])

# Esse valor será definido no main.py, onde o modelo é carregado
model_data = None


def get_user_profile(user_id: str):
    """
    Calcula o vetor de perfil do usuário com base nos artigos do seu histórico,
    extraídos do train_df. Para cada artigo do histórico, busca o índice na matriz TF-IDF
    (através de article_id_to_idx) e empilha os vetores para calcular a média.
    Retorna o vetor médio (como matriz 1xN).
    """
    # Filtra os registros do usuário no train_df
    user_rows = model_data['train_df'][model_data['train_df']['userId'] == user_id]
    if user_rows.empty:
        return None
    vectors = []
    for history in user_rows['history_list']:
        for article in history:
            idx = model_data['article_id_to_idx'].get(str(article))
            if idx is not None:
                vectors.append(model_data['tfidf_matrix'][idx])
    if not vectors:
        return None
    # Empilha os vetores (sparse) e calcula a média
    user_vector = vstack(vectors).mean(axis=0)
    return user_vector


@router.post("/", response_model=RecommendationOutput)
def get_recommendations(input_data: RecommendationInput):
    if model_data is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

    # Tenta obter o perfil do usuário
    user_vector = get_user_profile(input_data.user_id)

    # Se o perfil não for encontrado, utiliza o histórico fornecido no input
    if user_vector is None:
        user_input = " ".join(map(str, input_data.history))
        try:
            tfidf = model_data['tfidf']
            user_vector = tfidf.transform([user_input])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao processar histórico: {e}")

    # Converte o vetor para numpy array, caso seja np.matrix
    user_vector = np.asarray(user_vector)

    # Calcula a similaridade de cosseno
    similarities = cosine_similarity(user_vector, model_data['tfidf_matrix']).flatten()

    # Seleciona os índices dos 6 itens com maior similaridade
    top_indices = np.argsort(-similarities)[:6]

    # Se houver histórico enviado pelo usuário, evita recomendar itens já lidos
    user_history = set(map(str, input_data.history))
    recommendations = []
    for idx in top_indices:
        try:
            article = model_data['itens_df'].iloc[idx]
            article_id = str(article['Page'])
            if article_id in user_history:
                continue
            recommendations.append({
                'Page': article_id,
                'Title': article['Title']
            })
            if len(recommendations) >= 5:
                break
        except Exception:
            continue

    if not recommendations:
        raise HTTPException(status_code=404, detail="Nenhuma recomendação encontrada")

    return {"recommendations": recommendations}
