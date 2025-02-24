from fastapi import APIRouter, HTTPException
from app.models.data_models import RecommendationInput, RecommendationOutput
import numpy as np
import pandas as pd

router = APIRouter(prefix="/recommendation", tags=["Recommendation"])

model_data = None
lightfm_model_data = None


@router.post("/", response_model=RecommendationOutput)
def get_recommendations(input_data: RecommendationInput):
    # Se o usuário não tem histórico, utiliza o modelo TF-IDF/PLN
    if not input_data.history:
        try:
            recommendations = []
            itens_df = model_data.get('itens_df')
            if itens_df is None:
                raise Exception("DataFrame de itens não encontrado no modelo TF-IDF")

            # Ordena os itens pelo peso de recência (dos últimos 60 dias)
            sorted_df = itens_df.sort_values(by='recency_weight', ascending=False)
            for _, row in sorted_df.iterrows():
                recommendations.append({
                    'Page': row['Page'],
                    'Title': row['Title'] if pd.notnull(row['Title']) else "Título indisponível"
                })
                if len(recommendations) >= 5:
                    break

            if not recommendations:
                raise HTTPException(status_code=404, detail="Nenhuma recomendação encontrada")

            return {"recommendations": recommendations}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro no modelo TF-IDF: {str(e)}")

    # Se houver histórico, utiliza o fluxo do LightFM
    else:
        if lightfm_model_data is None:
            raise HTTPException(status_code=500, detail="Modelo LightFM não carregado")

        # Obtém o conjunto de notícias atuais (últimos 60 dias)
        valid_news = set(lightfm_model_data['itens_df']['Page'].astype(str).tolist())
        # Filtra o histórico para manter apenas itens presentes nesse conjunto
        filtered_history = [item for item in input_data.history if item in valid_news]

        user_mapping, dataset_item_mapping, _, _ = lightfm_model_data['lightfm_dataset'].mapping()
        user_id = input_data.user_id
        if user_id not in user_mapping:
            raise HTTPException(status_code=404, detail="Usuário não encontrado no modelo LightFM")

        user_idx = user_mapping[user_id]
        num_items = len(dataset_item_mapping)

        scores = lightfm_model_data['lightfm_model'].predict(
            user_idx,
            np.arange(num_items),
            item_features=lightfm_model_data['item_features']
        )

        # Expande para mais candidatos (top 20)
        top_indices = np.argsort(-scores)[:20]
        recommendations = []
        inverse_item_mapping = {v: k for k, v in dataset_item_mapping.items()}
        for idx in top_indices:
            item_id = inverse_item_mapping.get(idx)
            # Pula itens que não estejam entre as notícias atuais ou que já estejam no histórico filtrado
            if item_id not in valid_news or item_id in filtered_history:
                continue
            try:
                item_row = \
                lightfm_model_data['itens_df'][lightfm_model_data['itens_df']['Page'].astype(str) == item_id].iloc[0]
                title = item_row['Title']
            except Exception:
                title = "Título indisponível"
            recommendations.append({'Page': item_id, 'Title': title})
            if len(recommendations) >= 5:
                break

        # Se nenhuma recomendação válida foi gerada via LightFM, faz fallback para o modelo TF-IDF/PLN
        if not recommendations:
            try:
                recommendations = []
                itens_df = model_data.get('itens_df')
                if itens_df is None:
                    raise Exception("DataFrame de itens não encontrado no modelo TF-IDF")
                sorted_df = itens_df.sort_values(by='recency_weight', ascending=False)
                for _, row in sorted_df.iterrows():
                    # Opcional: pular itens já no histórico
                    if row['Page'] in filtered_history:
                        continue
                    recommendations.append({
                        'Page': row['Page'],
                        'Title': row['Title'] if pd.notnull(row['Title']) else "Título indisponível"
                    })
                    if len(recommendations) >= 5:
                        break
                if not recommendations:
                    raise HTTPException(status_code=404, detail="Nenhuma recomendação encontrada")
                return {"recommendations": recommendations}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Erro no modelo TF-IDF fallback: {str(e)}")

        return {"recommendations": recommendations}
