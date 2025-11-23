import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAMES_PATH = os.path.join(BASE_DIR, 'data', 'games.csv')
RATINGS_PATH = os.path.join(BASE_DIR, 'data', 'ratings.csv')


class RecommendationEngine:
    def __init__(self):
        self.games_df = pd.read_csv(GAMES_PATH)
        self.ratings_df = pd.read_csv(RATINGS_PATH)
        self.user_item_matrix = None
        self.svd = None
        self.item_similarity_matrix = None
        self._prepare_data()

    def _prepare_data(self):
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='user_id',
            columns='game_id',
            values='rating'
        ).fillna(0)

        n_components = min(15, self.user_item_matrix.shape[0] - 1, self.user_item_matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        user_matrix = self.svd.fit_transform(self.user_item_matrix)
        item_matrix = self.svd.components_.T
        self.reconstructed = np.dot(user_matrix, item_matrix.T)

        self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)

    def analyze_user(self, user_id):
        if user_id not in self.ratings_df['user_id'].values:
            return None

        user_data = self.ratings_df[self.ratings_df['user_id'] == user_id]
        user_games = user_data.merge(self.games_df, on='game_id')

        genre_analysis = user_games.groupby('genre').agg({
            'rating': ['count', 'mean', 'sum']
        }).reset_index()
        genre_analysis.columns = ['genre', 'count', 'avg_rating', 'total_score']
        genre_analysis = genre_analysis.sort_values('total_score', ascending=False)

        platform_counts = user_games['platform'].value_counts().to_dict()

        year_analysis = user_games.groupby('release_year')['rating'].agg(['count', 'mean']).reset_index()
        year_analysis.columns = ['year', 'count', 'avg_rating']

        rating_distribution = user_games['rating'].value_counts().sort_index().to_dict()

        top_games = user_games.nlargest(5, 'rating')[['title', 'genre', 'rating', 'release_year']].to_dict('records')

        return {
            'user_id': user_id,
            'total_rated': len(user_games),
            'average_rating': round(user_games['rating'].mean(), 2),
            'rating_std': round(user_games['rating'].std(), 2),
            'genre_stats': genre_analysis.to_dict('records'),
            'platform_stats': platform_counts,
            'year_stats': year_analysis.to_dict('records'),
            'rating_distribution': rating_distribution,
            'top_games': top_games,
            'favorite_genre': genre_analysis.iloc[0]['genre'] if len(genre_analysis) > 0 else 'N/A'
        }

    def get_svd_recommendations(self, user_id, n=10):
        if user_id not in self.user_item_matrix.index:
            return []

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        predictions = self.reconstructed[user_idx]

        rated_games = self.ratings_df[self.ratings_df['user_id'] == user_id]['game_id'].values

        recommendations = []
        for game_id, pred_score in enumerate(predictions, start=1):
            if game_id not in rated_games and game_id in self.games_df['game_id'].values:
                game = self.games_df[self.games_df['game_id'] == game_id].iloc[0]

                explanation = self._build_explanation(user_id, game_id, game, pred_score)

                recommendations.append({
                    'game_id': game_id,
                    'title': game['title'],
                    'genre': game['genre'],
                    'platform': game['platform'],
                    'year': game['release_year'],
                    'predicted_rating': round(pred_score, 2),
                    'confidence': min(int((pred_score / 5) * 100), 100),
                    'explanation': explanation
                })

        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:n]

    def _build_explanation(self, user_id, game_id, game_info, pred_score):
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        user_games = user_ratings.merge(self.games_df, on='game_id')

        same_genre = user_games[user_games['genre'] == game_info['genre']]
        same_platform = user_games[user_games['platform'] == game_info['platform']]

        factors = []

        if len(same_genre) > 0:
            genre_avg = same_genre['rating'].mean()
            genre_count = len(same_genre)
            factors.append({
                'category': 'Жанрова відповідність',
                'value': f"{genre_count} ігор жанру {game_info['genre']}",
                'detail': f"Середня оцінка: {genre_avg:.1f}",
                'weight': 'Висока'
            })

        if len(same_platform) > 0:
            platform_count = len(same_platform)
            factors.append({
                'category': 'Платформа',
                'value': f"{platform_count} ігор на {game_info['platform']}",
                'detail': f"Ви активно граєте на цій платформі",
                'weight': 'Середня'
            })

        similar_users = self.ratings_df[
            self.ratings_df['game_id'].isin(user_ratings['game_id'])
        ]['user_id'].unique()
        similar_users = [u for u in similar_users if u != user_id]

        if len(similar_users) > 0:
            similar_ratings = self.ratings_df[
                (self.ratings_df['user_id'].isin(similar_users)) &
                (self.ratings_df['game_id'] == game_id)
                ]
            if len(similar_ratings) > 0:
                similar_avg = similar_ratings['rating'].mean()
                factors.append({
                    'category': 'Collaborative Filtering',
                    'value': f"{len(similar_ratings)} схожих користувачів оцінили цю гру",
                    'detail': f"Їхня середня оцінка: {similar_avg:.1f}",
                    'weight': 'Висока'
                })

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_factors = self.svd.transform(self.user_item_matrix.iloc[[user_idx]])
        top_component = np.argmax(np.abs(user_factors[0]))
        factors.append({
            'category': 'Латентний фактор',
            'value': f"Фактор #{top_component + 1} (приховані переваги)",
            'detail': f"Сила впливу: {abs(user_factors[0][top_component]):.2f}",
            'weight': 'Середня'
        })

        main_reason = ""
        if len(same_genre) > 0 and same_genre['rating'].mean() >= 4:
            main_reason = f"Ви високо оцінюєте жанр {game_info['genre']} ({genre_avg:.1f} середня оцінка)"
        elif len(similar_ratings) > 0 and similar_avg >= 4:
            main_reason = f"Користувачі з схожими смаками високо оцінили цю гру ({similar_avg:.1f})"
        else:
            main_reason = f"Прогноз на основі аналізу вашого профілю та матричного розкладання"

        return {
            'main': main_reason,
            'factors': factors
        }

    def get_item_recommendations(self, game_title, n=10):
        if game_title not in self.games_df['title'].values:
            return []

        target_game = self.games_df[self.games_df['title'] == game_title].iloc[0]
        game_id = target_game['game_id']

        if game_id not in self.user_item_matrix.columns:
            return []

        game_idx = list(self.user_item_matrix.columns).index(game_id)
        similarities = self.item_similarity_matrix[game_idx]

        game_raters = self.ratings_df[self.ratings_df['game_id'] == game_id]['user_id'].unique()
        target_avg_rating = self.ratings_df[self.ratings_df['game_id'] == game_id]['rating'].mean()

        recommendations = []
        for idx, sim_score in enumerate(similarities):
            similar_game_id = self.user_item_matrix.columns[idx]

            if similar_game_id != game_id and similar_game_id in self.games_df['game_id'].values:
                game = self.games_df[self.games_df['game_id'] == similar_game_id].iloc[0]

                explanation = self._build_item_explanation(
                    target_game, game, similar_game_id, sim_score, game_raters, target_avg_rating
                )

                recommendations.append({
                    'game_id': similar_game_id,
                    'title': game['title'],
                    'genre': game['genre'],
                    'platform': game['platform'],
                    'year': game['release_year'],
                    'similarity': round(sim_score, 3),
                    'confidence': int(sim_score * 100),
                    'explanation': explanation
                })

        recommendations.sort(key=lambda x: x['similarity'], reverse=True)
        return recommendations[:n]

    def _build_item_explanation(self, target_game, similar_game, similar_id, sim_score, target_raters, target_avg):
        similar_raters = self.ratings_df[self.ratings_df['game_id'] == similar_id]['user_id'].unique()
        common_users = set(target_raters) & set(similar_raters)
        overlap_pct = (len(common_users) / len(target_raters)) * 100 if len(target_raters) > 0 else 0

        similar_avg_rating = self.ratings_df[self.ratings_df['game_id'] == similar_id]['rating'].mean()

        factors = []

        if len(common_users) > 0:
            common_ratings = self.ratings_df[
                (self.ratings_df['game_id'] == similar_id) &
                (self.ratings_df['user_id'].isin(common_users))
                ]
            common_avg = common_ratings['rating'].mean()

            factors.append({
                'category': 'Перетин аудиторії',
                'value': f"{len(common_users)} спільних гравців ({overlap_pct:.0f}%)",
                'detail': f"Їхня оцінка цієї гри: {common_avg:.1f}",
                'weight': 'Критична'
            })

        if similar_game['genre'] == target_game['genre']:
            factors.append({
                'category': 'Жанр',
                'value': f"Однаковий жанр: {similar_game['genre']}",
                'detail': f"Середня оцінка обох ігор близька ({target_avg:.1f} vs {similar_avg_rating:.1f})",
                'weight': 'Висока'
            })
        else:
            factors.append({
                'category': 'Жанр',
                'value': f"Різні жанри: {target_game['genre']} → {similar_game['genre']}",
                'detail': f"Але користувачі часто грають в обидві",
                'weight': 'Низька'
            })

        if similar_game['platform'] == target_game['platform']:
            factors.append({
                'category': 'Платформа',
                'value': f"Та сама платформа: {similar_game['platform']}",
                'detail': f"Доступність на тій же системі",
                'weight': 'Середня'
            })

        year_diff = abs(similar_game['release_year'] - target_game['release_year'])
        factors.append({
            'category': 'Період випуску',
            'value': f"Різниця: {year_diff} років",
            'detail': f"{target_game['release_year']} vs {similar_game['release_year']}",
            'weight': 'Низька' if year_diff > 5 else 'Середня'
        })

        factors.append({
            'category': 'Косинусна схожість',
            'value': f"Коефіцієнт: {sim_score:.3f}",
            'detail': f"Кореляція векторів оцінок користувачів",
            'weight': 'Технічна'
        })

        main_reason = ""
        if overlap_pct > 50:
            main_reason = f"{overlap_pct:.0f}% гравців {target_game['title']} також грали в цю гру"
        elif similar_game['genre'] == target_game['genre']:
            main_reason = f"Той самий жанр {similar_game['genre']} з високою кореляцією оцінок"
        else:
            main_reason = f"Високий коефіцієнт схожості за патернами оцінок ({sim_score:.2f})"

        return {
            'main': main_reason,
            'factors': factors
        }

    def get_metadata(self):
        return {
            "valid_ids": sorted(self.ratings_df['user_id'].unique().tolist()),
            "valid_games": sorted(self.games_df['title'].unique().tolist())
        }

    def get_statistics(self):
        total_games = len(self.games_df)
        total_users = self.ratings_df['user_id'].nunique()
        total_ratings = len(self.ratings_df)

        density = (total_ratings / (total_games * total_users)) * 100

        genre_dist = self.games_df['genre'].value_counts().to_dict()
        platform_dist = self.games_df['platform'].value_counts().to_dict()

        avg_rating = self.ratings_df['rating'].mean()
        rating_dist = self.ratings_df['rating'].value_counts().sort_index().to_dict()

        return {
            'total_games': total_games,
            'total_users': total_users,
            'total_ratings': total_ratings,
            'density': round(density, 2),
            'avg_rating': round(avg_rating, 2),
            'genres': genre_dist,
            'platforms': platform_dist,
            'rating_distribution': rating_dist
        }


engine = RecommendationEngine()


def get_user_analysis(user_id):
    return engine.analyze_user(user_id)


def get_svd_recommendations(user_id, n=10):
    return engine.get_svd_recommendations(user_id, n)


def get_item_recommendations(game_title, n=10):
    return engine.get_item_recommendations(game_title, n)


def get_system_metadata():
    return engine.get_metadata()


def get_system_statistics():
    return engine.get_statistics()