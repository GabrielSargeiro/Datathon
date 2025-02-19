import os

from fastapi import APIRouter, HTTPException
from app.models.data_models import RecommendationInput, RecommendationOutput
from app.services.model import load_model, recommend_from_model  # usaremos load_model e parte da lógica de recomendação
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

router = APIRouter(prefix="/recommendation", tags=["Recommendation"])

diretorio_modelo = os.path.dirname(os.getcwd())

# Carrega o modelo ao iniciar a aplicação
try:
    model_data = load_model()
except Exception as e:
    model_data = None
    print("Erro ao carregar o modelo:", e)


def get_user_profile(user_id: int):
    """
    Calcula o vetor de perfil do usuário como a média dos vetores das notícias lidas.
    Retorna None se não houver histórico ou ocorrer erro.
    """
    try:
        user_histories = model_data['train_df'][model_data['train_df']['userId'] == user_id]['history_list']
    except Exception as e:
        raise Exception(f"Erro ao acessar 'history_list' para userId {user_id}: {e}")

    article_ids = []
    for history in user_histories:
        try:
            article_ids.extend(history)
        except Exception as e:
            continue

    article_ids = list(set(article_ids))
    if not article_ids:
        return None

    indices = [model_data['article_id_to_idx'][str(a)]
               for a in article_ids if str(a) in model_data['article_id_to_idx']]
    if not indices:
        return None

    try:
        user_vector = model_data['tfidf_matrix'][indices].mean(axis=0)
    except Exception as e:
        raise Exception(f"Erro ao calcular a média dos vetores TF-IDF para userId {user_id}: {e}")

    return np.asarray(user_vector)


@router.post("/", response_model=RecommendationOutput)
def get_recommendations(input_data: RecommendationInput):
    if model_data is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

    # Tenta calcular o vetor do usuário usando o user_id
    user_vector = None
    try:
        user_vector = get_user_profile(input_data.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Se não foi possível calcular o vetor via perfil, utiliza o histórico enviado
    if user_vector is None:
        user_input = " ".join(map(str, input_data.history))
        try:
            tfidf = model_data['tfidf']
            user_vector = tfidf.transform([user_input])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao processar histórico: {e}")

    # Calcula similaridades entre o vetor do usuário e a matriz TF-IDF
    try:
        cosine_similarities = linear_kernel(user_vector, model_data['tfidf_matrix']).flatten()
        top_indices = cosine_similarities.argsort()[-6:][::-1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao calcular similaridades: {e}")

    recommendations = []
    if input_data.user_id:
        # Se usou o perfil, evite recomendar artigos já lidos
        user_history = set()
        try:
            for hist in model_data['train_df'][model_data['train_df']['userId'] == input_data.user_id]['history_list']:
                try:
                    user_history.update(map(str, hist))
                except Exception as e:
                    continue
        except Exception as e:
            # Caso não consiga extrair o histórico, apenas prossiga
            pass

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
            except Exception as e:
                continue
    else:
        for idx in top_indices[:5]:
            try:
                article = model_data['itens_df'].iloc[idx]
                recommendations.append({
                    'Page': str(article['Page']),
                    'Title': article['Title']
                })
            except Exception as e:
                continue

    if not recommendations:
        raise HTTPException(status_code=404, detail="Nenhuma recomendação encontrada")

    return {"recommendations": recommendations}
