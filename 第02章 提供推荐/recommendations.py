from itertools import combinations
from math import pow, sqrt
from typing import Dict, Callable, List, Tuple

# 一个涉及影评者及其对几部影片评分情况的字典
critics = {'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
                         'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
                         'The Night Listener': 3.0},
           'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
                            'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
                            'You, Me and Dupree': 3.5},
           'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
                                'Superman Returns': 3.5, 'The Night Listener': 4.0},
           'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
                            'The Night Listener': 4.5, 'Superman Returns': 4.0,
                            'You, Me and Dupree': 2.5},
           'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                            'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
                            'You, Me and Dupree': 2.0},
           'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                             'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
           'Toby': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0}}
MOVIE_DATA = Dict[str, Dict[str, float]]


def sim_distance(
        prefs: MOVIE_DATA,
        person_1: str,
        person_2: str
) -> float:
    """欧几里得距离评价"""
    person_1_data = prefs.get(person_1, {})
    person_2_data = prefs.get(person_2, {})
    # 如果没有这个人，当作异常处理
    if not person_1 or not person_2:
        return -1.0
    # 获取共同爱好的 keys
    common_movies = person_1_data.keys() & person_2_data.keys()
    # 如果没有共同爱好
    if not common_movies:
        return .0
    # 求平方和
    sum_of_squares = sum(
        map(lambda movie_name: pow(person_1_data[movie_name] - person_2_data[movie_name], 2), common_movies))

    return 1 / (sqrt(sum_of_squares) + 1)


def sim_pearson(
        prefs: MOVIE_DATA,
        person_1: str,
        person_2: str
) -> float:
    """皮尔逊相关度"""
    person_1_data = prefs.get(person_1, {})
    person_2_data = prefs.get(person_2, {})
    # 获取共同爱好的 keys
    common_movies = person_1_data.keys() & person_2_data.keys()
    if not common_movies:
        return 0
    n = len(common_movies)
    # 求和偏好分
    sum_p1 = sum((person_1_data[movie] for movie in common_movies))
    sum_p2 = sum((person_2_data[movie] for movie in common_movies))

    # 求平方和偏好分
    sum_pow_p1 = sum(((pow(person_1_data[movie], 2) for movie in common_movies)))
    sum_pow_p2 = sum(((pow(person_2_data[movie], 2) for movie in common_movies)))

    # 求评分乘积和
    p_sum = sum((person_1_data[movie] * person_2_data[movie] for movie in common_movies))

    # 计算皮尔逊评价

    num = p_sum - (sum_p1 * sum_p2 / n)
    den = sqrt((sum_pow_p1 - pow(sum_p1, 2) / n) * (sum_pow_p2 - pow(sum_p2, 2) / n))
    if den == 0:
        return 0
    r = num / den
    return r


# 计算两两的相似度
for p1, p2 in combinations(critics, 2):
    sim_point = sim_distance(critics, p1, p2)
    print(f"{sim_distance.__name__} - {p1} with {p2}: {sim_point}")

    sim_point = sim_pearson(critics, p1, p2)
    print(f"{sim_pearson.__name__} - {p1} with {p2}: {sim_point}")


def to_matches(
        prefs: MOVIE_DATA,
        person: str,
        n=5,
        similarity: Callable = sim_pearson
) -> List[Tuple[float, str]]:
    """找到相似的人"""
    res = [(similarity(prefs, person, o_user), o_user) for o_user in prefs.keys() if o_user != person]
    res.sort(reverse=True)
    return res[:n]


def get_recommendations(
        prefs: MOVIE_DATA,
        person: str,
        similarity: Callable = sim_pearson
) -> List[Tuple[float, str]]:
    """获取推荐的电影"""
    res = {}
    for other in prefs.keys():
        if other == person:
            continue
        sim = similarity(prefs, person, other)
        # 忽略不相似的人
        if sim <= 0:
            continue
        # 遍历影片
        looked_movies = prefs[person].keys()
        for movie in prefs[other]:
            # 只安排自己没评价过的电影
            if movie not in looked_movies or prefs[person][movie] == 0:
                res.setdefault(movie, [0, 0])
                res[movie][0] += sim * prefs[other][movie]
                res[movie][1] += sim
    # 归一化
    rankings = [(sim_sum / n_sum, movie) for movie, (sim_sum, n_sum) in res.items()]
    rankings.sort(reverse=True)
    return rankings


def transform_prefs(prefs: MOVIE_DATA) -> MOVIE_DATA:
    """对调人和电影"""
    res = {}
    for person, movie_items in prefs.items():
        for movie_name, point in movie_items.items():
            res.setdefault(movie_name, {})
            res[movie_name][person] = point
    return res


def calculate_similar_items(prefs: MOVIE_DATA, n: int = 10) -> Dict[str, List]:
    """计算相似物品之间的相似度"""
    res = {}
    items_prefs = transform_prefs(prefs)
    for item in items_prefs:
        scores = to_matches(items_prefs, item, n=n, similarity=sim_distance)
        res[item] = scores
    return res


def get_recommend_items(prefs: MOVIE_DATA, item_match: Dict[str, List], user: str):
    user_data = prefs[user]
    scores_items = {}
    for item, rating in user_data.items():
        for similarity, s_item in item_match[item]:
            if s_item in user_data:
                continue
            scores_items.setdefault(item, [0, 0])
            scores_items[item][0] += rating * similarity
            scores_items[item][1] += similarity

    rankings = [(weighted_sum / t_sum, item) for item, (weighted_sum, t_sum) in scores_items.items()]
    rankings.sort(reverse=True)
    return rankings


def load_movie_lens(path: str = '../data/movielens') -> MOVIE_DATA:
    # 获取影片标题
    movies = {}
    with open(path + "/u.item", encoding="u8") as fp:
        for text in fp:
            m_id, title, *_ = text.split("|")
            movies[m_id] = title
    # 获取用户数据
    prefs = {}
    with open(path + "/u.data") as fp:
        for text in fp:
            user, m_id, rating, *_ = text.split('\t')
            prefs.setdefault(user, {})
            prefs[user][movies[m_id]] = float(rating)

    return prefs


if __name__ == '__main__':
    prefs = load_movie_lens()
