from fastapi import APIRouter, HTTPException
from app.models.data_models import RecommendationInput, RecommendationOutput
import numpy as np

router = APIRouter(prefix="/recommendation", tags=["Recommendation"])

model_data = None

@router.post("/", response_model=RecommendationOutput)
def get_recommendations(input_data: RecommendationInput):
    if model_data is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

    # Converte o user_id (externo) para o índice interno usado no LightFM
    user_external = input_data.user_id
    user_internal = model_data['user_mapping'].get(user_external)
    if user_internal is None:
        raise HTTPException(status_code=404, detail="Usuário não encontrado no mapeamento do modelo")

    # Obtém o total de itens (supondo que model_data['item_mapping'] seja um dict com índice interno como chave)
    num_items = len(model_data['item_mapping'])

    # Gera scores para todos os itens usando o método predict do LightFM
    scores = model_data['model'].predict(user_internal, np.arange(num_items))

    # Seleciona os top 6 itens com maior score
    top_indices = np.argsort(-scores)[:6]

    # Se houver histórico de itens já lidos, vamos evitá-los
    user_history = set()
    try:
        for hist in model_data['train_df'][model_data['train_df']['userId'] == user_external]['history_list']:
            try:
                user_history.update(map(str, hist))
            except Exception:
                continue
    except Exception:
        pass

    recommendations = []
    for idx in top_indices:
        try:
            # Recupera o item correspondente a partir do mapeamento
            # Supondo que model_data['item_mapping'] seja um dict {índice_interno: item_id}
            item_id = model_data['item_mapping'].get(idx)
            if item_id is None:
                continue

            # Se o item já foi lido, ignora
            if str(item_id) in user_history:
                continue

            # Busca os dados do item (por exemplo, Page e Title) no DataFrame de itens
            article = model_data['itens_df'][model_data['itens_df']['Page'] == item_id].iloc[0]
            recommendations.append({
                'Page': str(article['Page']),
                'Title': article['Title']
            })
            if len(recommendations) >= 5:
                break
        except Exception:
            continue

    if not recommendations:
        raise HTTPException(status_code=404, detail="Nenhuma recomendação encontrada")

    return {"recommendations": recommendations}
